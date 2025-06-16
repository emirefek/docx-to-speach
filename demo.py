import os
import shutil
import argparse
import json
import tempfile
from docx import Document
from pydub import AudioSegment
from tqdm import tqdm

# --- Configuration ---
# !!! IMPORTANT: YOU MUST MODIFY 'speaker_wav_path' to your actual speaker audio file.
# !!! You might also want to change 'default_docx_path' for easier testing without CLI args.
CONFIG = {
    "speaker_wav_path": "female.wav",  # REQUIRED: Update this path
    "language": "tr",  # Language for XTTS (e.g., "en", "es", "fr")
    "xtts_model_name": "tts_models/multilingual/multi-dataset/xtts_v2", # Recommended XTTS model
    "device": "cpu",  # For Apple Silicon (M1, M2, M3), use "mps". Use "cuda" for NVIDIA, "cpu" otherwise.
    "temp_dir_name": ".temp_audio_segments_xtts",
    "default_output_filename": "generated_speech_xtts.wav",
    "default_docx_path": "input.docx" # A default DOCX path for quick testing if no CLI arg is given
}
# --- End Configuration ---

def ensure_ffmpeg_is_installed():
    """Checks for ffmpeg using pydub's method, which is indirect but practical."""
    try:
        AudioSegment.silent() # This often relies on ffmpeg for creating silence
        # A more direct check could be shutil.which("ffmpeg")
        if not shutil.which("ffmpeg"):
            print("WARNING: ffmpeg not found in PATH. pydub may fail.")
            print("Please install ffmpeg and ensure it's in your system PATH.")
            print("On macOS with Homebrew: brew install ffmpeg")
            return False
        return True
    except Exception as e:
        print(f"Error during ffmpeg check (or pydub issue): {e}")
        print("Please ensure ffmpeg is installed and accessible by pydub.")
        print("On macOS with Homebrew: brew install ffmpeg")
        return False

def create_tts_instance(model_name, device):
    """Initializes and returns the TTS object."""
    try:
        import torch # Ensure torch is imported
        from TTS.api import TTS as CoquiTTS
        
        # --- Workaround for PyTorch 2.6 weights_only=True issue with XTTS ---
        # Attempt to allowlist the XttsConfig class
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
                print("Attempting to apply PyTorch 2.6 workaround: add_safe_globals for XttsConfig")
                torch.serialization.add_safe_globals([XttsConfig])
            else:
                # This might be an older PyTorch version where this isn't needed, or the API changed
                print("torch.serialization.add_safe_globals not found, proceeding without it (may not be needed for this PyTorch version).")

        except ImportError:
            print("WARNING: Could not import XttsConfig for PyTorch 2.6 workaround. This might be an issue if using PyTorch >= 2.6.")
        except Exception as e_sg:
            print(f"WARNING: Error applying add_safe_globals workaround: {e_sg}")
        # --- End Workaround ---

        print(f"Initializing Coqui TTS with model: {model_name} on device: {device}")
        tts_instance = CoquiTTS(model_name=model_name, progress_bar=True)
        tts_instance.to(device) # Move model to the specified device
        print("Coqui TTS initialized successfully.")
        return tts_instance
    except ModuleNotFoundError:
        print("ERROR: The 'TTS' module from Coqui AI was not found.")
        print("Please install it by running: pip install TTS")
        return None
    except Exception as e:
        print(f"ERROR: Could not initialize Coqui TTS model: {e}")
        print("This might be due to model download issues, incorrect model name, or device compatibility.")
        if "weights_only" in str(e) and "PyTorch 2.6" in str(e): # Be more specific if possible
             print("This looks like the PyTorch 2.6 compatibility issue with XTTS.")
             print("If the workaround didn't help, consider downgrading PyTorch, e.g.:")
             print("pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu  (for CPU)")
             print("Or check PyTorch website for MPS compatible version of 2.5.1 if available, or a nightly build of TTS.")
        if device == "mps" and "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
             print("For MPS issues, you might try setting: export PYTORCH_ENABLE_MPS_FALLBACK=1")
        return None

def extract_paragraphs_from_docx(docx_path):
    """Extracts all paragraphs from a .docx file."""
    try:
        doc = Document(docx_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        if not paragraphs:
            print(f"WARNING: No text found in DOCX file: {docx_path}")
        return paragraphs
    except Exception as e:
        print(f"ERROR: Could not read or parse DOCX file '{docx_path}': {e}")
        return []

def synthesize_text_to_file(tts_instance, text, speaker_wav, language, output_path):
    """Synthesizes text to an audio file using the given TTS instance."""
    try:
        tts_instance.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            file_path=output_path,
        )
        return True
    except Exception as e:
        print(f"ERROR: Coqui TTS synthesis failed for text segment: '{text[:50]}...'")
        print(f"Details: {e}")
        # Check if speaker_wav exists
        if not os.path.exists(speaker_wav):
            print(f"CRITICAL ERROR: Speaker WAV file not found at '{speaker_wav}'. Please check CONFIG.")
        return False

def combine_audio_files(segment_paths, output_path):
    """Combines multiple WAV files into a single WAV file using pydub."""
    if not segment_paths:
        print("WARNING: No audio segments to combine.")
        return False
    try:
        combined = AudioSegment.empty()
        print("Combining audio segments...")
        for segment_path in tqdm(segment_paths, desc="Combining Audio"):
            segment_audio = AudioSegment.from_wav(segment_path)
            combined += segment_audio
        
        combined.export(output_path, format="wav")
        print(f"Successfully combined audio into: {output_path}")
        return True
    except Exception as e:
        print(f"ERROR: Could not combine audio files: {e}")
        return False

def main():
    if not ensure_ffmpeg_is_installed():
        # Allow script to continue but pydub parts will likely fail
        print("Continuing without confirmed ffmpeg, but audio combination/export might fail.")

    parser = argparse.ArgumentParser(description="Convert DOCX file to speech using Coqui XTTS.")
    parser.add_argument("--docx", type=str, default=CONFIG["default_docx_path"],
                        help=f"Path to the input .docx file (default: {CONFIG['default_docx_path']})")
    parser.add_argument("--output", type=str, default=CONFIG["default_output_filename"],
                        help=f"Path to the output .wav file (default: {CONFIG['default_output_filename']})")
    parser.add_argument("--speaker_wav", type=str, default=CONFIG["speaker_wav_path"],
                        help=f"Path to the speaker reference .wav file (default: from config)")
    parser.add_argument("--language", type=str, default=CONFIG["language"],
                        help=f"Language code for TTS (e.g., 'en', 'es') (default: {CONFIG['language']})")
    parser.add_argument("--device", type=str, default=CONFIG["device"],
                        help=f"Device for TTS ('mps', 'cuda', 'cpu') (default: {CONFIG['device']})")
    
    args = parser.parse_args()

    # Update config with CLI args if they are different from defaults or specifically provided
    # This allows overriding config speaker_wav, language, device via CLI
    current_speaker_wav = args.speaker_wav
    current_language = args.language
    current_device = args.device
    docx_file_path = args.docx
    output_file_path = args.output

    if not os.path.exists(current_speaker_wav):
        print(f"ERROR: Speaker WAV file not found at '{current_speaker_wav}'. Please check the path.")
        print("Update the CONFIG in the script or provide a valid path using --speaker_wav.")
        return

    if not os.path.exists(docx_file_path):
        print(f"ERROR: DOCX file not found at '{docx_file_path}'. Please check the path.")
        return

    tts = create_tts_instance(CONFIG["xtts_model_name"], current_device)
    if not tts:
        return

    paragraphs = extract_paragraphs_from_docx(docx_file_path)
    if not paragraphs:
        print("No paragraphs extracted or DOCX file was empty/unreadable. Exiting.")
        return

    # Create a temporary directory for audio segments
    #temp_dir = os.path.join(os.path.dirname(output_file_path) or '.', CONFIG["temp_dir_name"]) # place temp next to output
    temp_dir = tempfile.mkdtemp(prefix="docx_to_speech_") # Use system temp
    print(f"Using temporary directory for audio segments: {temp_dir}")
    
    os.makedirs(temp_dir, exist_ok=True)
    
    audio_segment_paths = []
    has_errors = False

    print(f"\nSynthesizing {len(paragraphs)} paragraphs...")
    for i, p_text in enumerate(tqdm(paragraphs, desc="Synthesizing Paragraphs")):
        if not p_text.strip(): # Skip empty paragraphs
            continue
        segment_filename = os.path.join(temp_dir, f"segment_{i:04d}.wav")
        if synthesize_text_to_file(tts, p_text, current_speaker_wav, current_language, segment_filename):
            audio_segment_paths.append(segment_filename)
        else:
            print(f"Skipping paragraph {i+1} due to synthesis error.")
            has_errors = True # Mark that an error occurred, but continue processing others

    if not audio_segment_paths:
        print("No audio segments were successfully synthesized. Cannot create final audio file.")
    else:
        combine_audio_files(audio_segment_paths, output_file_path)

    # Cleanup temporary directory
    try:
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)
        print("Temporary directory cleaned up.")
    except Exception as e:
        print(f"WARNING: Could not clean up temporary directory '{temp_dir}': {e}")
        print("You may need to delete it manually.")

    if has_errors:
        print("\nProcess completed with some errors during paragraph synthesis.")
    elif not audio_segment_paths and paragraphs: # Had paragraphs but no audio
        print("\nProcess completed, but no audio could be generated.")
    elif not paragraphs:
         print("\nProcess completed, no text found in document.")
    else:
        print("\nProcess completed successfully!")

if __name__ == "__main__":
    main()