"""
Easy-to-use transcription entry point.

Examples
--------
    venv\\Scripts\\python.exe inference.py --transcribe audio.wav
    venv\\Scripts\\python.exe inference.py --transcribe audio_folder

Implementation lives in ``src.cli.inference``. This root file is intentionally
only a launcher so the user-facing command remains short without putting
application logic outside the package architecture.
"""

from src.cli.inference import main


if __name__ == "__main__":
    main()