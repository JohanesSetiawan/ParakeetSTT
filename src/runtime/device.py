"""
Centralized device resolution and runtime environment reporting.

Every component receives the device chosen here instead of probing hardware on
its own, so the model, feature extractor, and reporting always agree.

Device selection and synchronization go through ``torch.accelerator``, the
device-agnostic PyTorch API: whatever accelerator the installed PyTorch build
supports and the machine has (CUDA, ROCm, Apple MPS, Intel XPU, ...) is used,
and the CPU otherwise. Features that exist only on NVIDIA GPUs (tensor-core
float16, CUDA Graphs, the allocator cap) check for CUDA themselves.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import statistics
import time
from dataclasses import dataclass

import torch
from torch.nn import functional as torch_functional

logger = logging.getLogger(__name__)

# Tensor cores, which make float16 fast, arrived with compute capability 7.0
# (Volta). Older GPUs (Pascal and before) run float16 arithmetic slowly or not
# at all, so "auto" keeps them in float32.
TENSOR_CORE_CAPABILITY = (7, 0)

# The Turing chips TU116 and TU117 have no tensor cores but the same compute
# capability (7.5) as the RTX 20 series, which has them; only the product
# name tells them apart: GeForce GTX 16xx, GeForce MX450 and MX550, and the
# T400, T500, T550, T600, T1000, T1200, and T2000 workstation cards.
TURING_CAPABILITY = (7, 5)
TURING_WITHOUT_TENSOR_CORES = re.compile(r"GTX 16\d\d|\bMX[45]50\b|\bT(?:400|500|550|600|1000|1200|2000)\b")

# float16 accumulation is turned on by "auto" only when it makes the
# encoder's matrix products at least this much faster on the GPU in use. The
# RTX 3050 Ti measured 1.68 to 1.84 times across runs; a GPU that already
# accumulates float32 at full rate is expected near 1.0. A threshold far from
# both keeps the decision, and with it the transcripts, the same from run to
# run on one GPU instead of flipping with clock speed.
FLOAT16_ACCUMULATION_MIN_SPEEDUP = 1.25

# Rows of the float16 accumulation microbenchmark: 16 full chunks of 188
# encoder frames, the encoder's largest matrix products at the default batch.
MICROBENCHMARK_TOKENS = 16 * 188
MICROBENCHMARK_ROUNDS = 3
MICROBENCHMARK_REPEATS = 10

# Decode threads per logical CPU for decode_workers = "auto": 4 threads on
# the 16 logical CPUs of the reference laptop decoded fastest (raw MP3
# decoding 4.4 s on 1 thread, 3.1 s on 2, 2.3 s on 4, 2.5 s on 8).
LOGICAL_CPUS_PER_DECODE_WORKER = 4
MAX_AUTO_DECODE_WORKERS = 8


@dataclass(frozen=True)
class RuntimeReport:
    """Dynamically detected software and hardware facts for one process."""

    device: str
    accelerator: str
    device_count: int
    device_name: str
    compute_capability: str
    precision: str
    torch_version: str
    cuda_version: str
    python_version: str

    @property
    def accelerator_label(self) -> str:
        """The accelerator for people: ``CPU`` when there is none."""

        return "CPU" if self.accelerator == "none" else self.accelerator

    def summary(self, precision_description: str) -> str:
        """
        Everything ``lines`` reports, on one line for the terminal.

        Args:
            precision_description: How the model runs, e.g. "float16 encoder,
                float32 decoder and joint".
        """

        devices = f"{self.device_count} device" + ("s" if self.device_count != 1 else "")
        return (
            f"Running on: {self.device_name} ({self.accelerator_label}, {devices}), "
            f"{precision_description}, PyTorch {self.torch_version}, CUDA {self.cuda_version}"
        )

    def lines(self) -> tuple[str, ...]:
        """Return plain-text ``Label: value`` lines for terminal and log."""

        return (
            f"Device: {self.device}",
            f"Accelerator: {self.accelerator}",
            f"Device count: {self.device_count}",
            f"Device name: {self.device_name}",
            f"Compute capability: {self.compute_capability}",
            f"Precision: {self.precision}",
            f"PyTorch: {self.torch_version}",
            f"CUDA runtime: {self.cuda_version}",
            f"Python: {self.python_version}",
        )


# =============================================================================
# Device selection and synchronization
# =============================================================================


def select_device() -> torch.device:
    """
    The accelerator PyTorch can use on this machine, else the CPU.

    ``torch.accelerator`` knows which accelerator the installed build was
    made for; ``check_available`` also requires it to be usable now (a CUDA
    build on a machine without an NVIDIA GPU returns None here).
    """

    accelerator = torch.accelerator.current_accelerator(check_available=True)
    if accelerator is None:
        return torch.device("cpu")
    return torch.device(accelerator.type)


def synchronize(device: torch.device) -> None:
    """Wait for the device's queued work; a no-op on the CPU."""

    if device.type != "cpu":
        torch.accelerator.synchronize(device)


# What device_family returns for the two kinds of "cuda" device.
NVIDIA_FAMILY = "NVIDIA"
ROCM_FAMILY = "ROCm"


def device_family(device: torch.device) -> str:
    """
    Short name of the device for messages.

    ``NVIDIA_FAMILY`` for NVIDIA GPUs, ``ROCM_FAMILY`` for AMD GPUs (which
    PyTorch also calls "cuda"), otherwise the device type ("cpu", "mps",
    "xpu", ...).
    """

    if is_nvidia_gpu(device):
        return NVIDIA_FAMILY
    if device.type == "cuda":
        return ROCM_FAMILY
    return device.type


def is_nvidia_gpu(device: torch.device) -> bool:
    """
    Whether the device is an NVIDIA GPU.

    ROCm builds of PyTorch present AMD GPUs as "cuda" devices too; their
    compute capability and architecture names mean something else.
    """

    return device.type == "cuda" and torch.version.hip is None


def cuda_capability(device: torch.device) -> tuple[int, int] | None:
    """The NVIDIA compute capability, or None for any other device."""

    if not is_nvidia_gpu(device):
        return None
    return torch.cuda.get_device_capability(device)


def has_tensor_cores(device: torch.device) -> bool:
    """Whether the device is an NVIDIA GPU with tensor cores (fast float16)."""

    capability = cuda_capability(device)
    if capability is None or capability < TENSOR_CORE_CAPABILITY:
        return False
    if capability == TURING_CAPABILITY and TURING_WITHOUT_TENSOR_CORES.search(torch.cuda.get_device_name(device)):
        return False
    return True


def build_supports_device(device: torch.device) -> bool | None:
    """
    Whether the installed PyTorch build carries kernels for this CUDA GPU.

    Each build ships machine code for a list of architectures (``sm_XY``) and
    portable code (``compute_XY``) that newer GPUs can compile at load time.
    A GPU older than every entry is not supported by the build at all.

    Returns:
        True or False for an NVIDIA GPU, None for other devices (ROCm builds
        list AMD architecture names, which this check does not interpret).
    """

    capability = cuda_capability(device)
    if capability is None:
        return None
    major, minor = capability
    architectures = torch.cuda.get_arch_list()
    if f"sm_{major}{minor}" in architectures:
        return True
    # Same major version, lower minor: binary compatible. Portable compute_XY
    # code runs on any GPU at least that new.
    for architecture in architectures:
        kind, _, version = architecture.partition("_")
        if not version.isdigit() or len(version) < 2:
            continue
        built = (int(version[:-1]), int(version[-1]))
        if kind == "sm" and built[0] == major and built[1] <= minor:
            return True
        if kind == "compute" and built <= capability:
            return True
    return False


# =============================================================================
# Precision
# =============================================================================


def encoder_dtype_for(precision: str, device: torch.device) -> tuple[torch.dtype, str]:
    """
    Resolve the configured encoder precision for the selected device.

    "auto" uses float16 on NVIDIA GPUs with tensor cores and float32
    everywhere else. An explicit "float16" is used on any NVIDIA GPU (with a
    warning when it has no tensor cores); other devices run float32, because
    CPUs lack fast float16 kernels and AMD (ROCm), MPS, and other
    accelerators have not been verified.

    Returns:
        The dtype and a one-line explanation for the terminal and log.
    """

    capability = cuda_capability(device)
    capability_text = f"compute capability {capability[0]}.{capability[1]}" if capability else ""

    if precision == "auto":
        if has_tensor_cores(device):
            return torch.float16, f"float16 encoder, float32 decoder and joint (auto: tensor cores, {capability_text})"
        if is_nvidia_gpu(device):
            return torch.float32, f"float32 (auto: no tensor cores, {capability_text})"
        return torch.float32, f"float32 (auto: float16 is used on NVIDIA GPUs only, this is {device_family(device)})"

    if precision == "float16" and is_nvidia_gpu(device):
        if not has_tensor_cores(device):
            logger.warning("encoder_precision float16 on a GPU without tensor cores (%s)", capability_text)
            return torch.float16, f"float16 encoder, float32 decoder and joint (no tensor cores, {capability_text})"
        return torch.float16, "float16 encoder, float32 decoder and joint"
    if precision == "float16":
        return torch.float32, f"float32 (float16 is used on NVIDIA GPUs only, this is {device_family(device)})"
    return torch.float32, "float32"


def apply_float32_matmul_precision(precision: str) -> None:
    """
    Set how float32 matrix products run, for the whole process.

    "high" lets GPUs with TF32 tensor cores (NVIDIA Ampere and newer) use
    them: a float32 encoder ran 28 percent faster with the same tokens on the
    sampled batches. With a float16 encoder only the decoder and joint use
    float32 products, and those are bound by reading weights, not arithmetic.
    """

    torch.set_float32_matmul_precision(precision)


def float16_accumulation_supported() -> bool:
    """Whether this PyTorch build has the cuBLAS float16 accumulation switch."""

    return hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation")


def apply_float16_accumulation(enabled: bool, encoder_dtype: torch.dtype) -> str:
    """
    Let cuBLAS accumulate float16 matrix products in float16, for the whole process.

    Tensor cores of consumer GPUs run float16 products about twice as fast
    when they also add in float16. Only the float16 encoder is affected (the
    decoder and joint are float32). On the RTX 3050 Ti the encoder ran 1.17
    times faster, and the word error rate on the docs corpus was 6.66 percent
    against 6.67 with float32 accumulation. The flag is set explicitly either
    way, so a process never inherits it from earlier code.

    Returns:
        A one-line state for the terminal and log.
    """

    supported = float16_accumulation_supported()
    active = enabled and encoder_dtype == torch.float16 and supported
    if supported:
        torch.backends.cuda.matmul.allow_fp16_accumulation = active

    if active:
        return "on"
    if not enabled:
        return "off"
    if encoder_dtype != torch.float16:
        return "off (the encoder runs in float32)"
    logger.warning("float16 accumulation requested but this PyTorch build does not support it")
    return "off (not supported by this PyTorch build)"


def measure_float16_accumulation_speedup(
    device: torch.device,
    hidden_size: int,
    intermediate_size: int,
) -> float | None:
    """
    How much faster the encoder's feed-forward products run with float16 accumulation.

    Times the two feed-forward matrix products (``hidden -> intermediate`` and
    back) on random float16 data, alternating the switch off and on so clock
    changes cannot favor one side. Takes well under a second. The switch is
    restored to its previous state.

    Returns:
        Time with float32 accumulation divided by time with float16
        accumulation, or None when it cannot be measured (not an NVIDIA GPU,
        or a build without the switch).
    """

    if not is_nvidia_gpu(device) or not float16_accumulation_supported():
        return None

    matmul_backend = torch.backends.cuda.matmul
    previous = matmul_backend.allow_fp16_accumulation
    generator = torch.Generator(device=device).manual_seed(0)
    options = {"dtype": torch.float16, "device": device, "generator": generator}
    activations = torch.randn(MICROBENCHMARK_TOKENS, hidden_size, **options)
    expansion = torch.randn(intermediate_size, hidden_size, **options)
    projection = torch.randn(hidden_size, intermediate_size, **options)

    def feed_forward() -> None:
        torch_functional.linear(torch_functional.linear(activations, expansion), projection)

    def seconds_with(flag: bool) -> float:
        matmul_backend.allow_fp16_accumulation = flag
        feed_forward()  # cuBLAS picks its kernel for this setting
        synchronize(device)
        started = time.perf_counter()
        for _ in range(MICROBENCHMARK_REPEATS):
            feed_forward()
        synchronize(device)
        return time.perf_counter() - started

    try:
        samples: dict[bool, list[float]] = {False: [], True: []}
        for _ in range(MICROBENCHMARK_ROUNDS):
            for flag in (False, True):
                samples[flag].append(seconds_with(flag))
    finally:
        matmul_backend.allow_fp16_accumulation = previous
    return statistics.median(samples[False]) / statistics.median(samples[True])


def default_decode_workers(logical_cpus: int | None = None) -> int:
    """
    Decode threads for ``decode_workers = "auto"``: one per four logical CPUs, 1 to 8.

    Args:
        logical_cpus: CPUs to plan for; ``os.cpu_count()`` when omitted.
    """

    cpus = logical_cpus if logical_cpus is not None else (os.cpu_count() or 1)
    return max(1, min(MAX_AUTO_DECODE_WORKERS, cpus // LOGICAL_CPUS_PER_DECODE_WORKER))


# =============================================================================
# Runtime report
# =============================================================================


def _accelerator_device_name(device: torch.device) -> str:
    """A readable device name from the backend module, when it offers one."""

    if device.type == "mps":
        return "Apple Metal"
    backend = torch.get_device_module(device.type)
    get_name = getattr(backend, "get_device_name", None)
    if get_name is None:
        return device.type
    return str(get_name(device.index if device.index is not None else torch.accelerator.current_device_index()))


def describe_runtime(device: torch.device, dtype: torch.dtype) -> RuntimeReport:
    """
    Describe the resolved device and library versions.

    Args:
        device: Device the model parameters live on.
        dtype: Parameter dtype actually used for inference.

    Returns:
        Report whose values are all read from the running process.
    """

    if device.type == "cpu":
        accelerator = "none"
        device_count = 1
        device_name = platform.processor() or platform.machine()
    else:
        if device_family(device) == ROCM_FAMILY:
            accelerator = ROCM_FAMILY
        else:
            accelerator = device.type.upper()
        device_count = torch.accelerator.device_count()
        device_name = _accelerator_device_name(device)

    capability = cuda_capability(device)
    return RuntimeReport(
        device=str(device),
        accelerator=accelerator,
        device_count=device_count,
        device_name=device_name,
        compute_capability=f"{capability[0]}.{capability[1]}" if capability else "not applicable",
        precision=str(dtype).removeprefix("torch."),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda or "not available",
        python_version=platform.python_version(),
    )
