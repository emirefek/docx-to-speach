import os
import shutil
import argparse
import tempfile
from docx import Document
from pydub import AudioSegment
from tqdm import tqdm
import torch
import sys # For sys.exit()

# Attempt to import nltk
try:
    import nltk
except ImportError:
    print("ERROR: 'nltk' library is not installed. Please install it by running: pip install nltk")
    print("The script will attempt to download its 'punkt' resource if nltk is found, but sentence splitting will be affected if nltk is missing.")
    nltk = None # Set to None so later checks can skip nltk-dependent parts

# --- Import Configuration ---
try:
    import config as app_config # Assuming config.py is in the same directory
except ImportError:
    print("CRITICAL ERROR: config.py not found.")
    print("This script requires config.py to be present in the same directory and properly configured.")
    print("Please create or restore config.py before running the script.")
    sys.exit(1) # Exit the script with an error code
# --- End Import Configuration ---


def ensure_ffmpeg_is_installed():
    """Checks for ffmpeg using pydub's method or shutil.which."""
    try:
        if shutil.which("ffmpeg"): # More direct check
            return True
        AudioSegment.silent() # pydub's indirect check
        print("ffmpeg found via pydub's check (indirect).") # Should not be reached if shutil.which works
        return True
    except Exception: # Broad exception as pydub might raise various things if ffmpeg is missing
        print("WARNING: ffmpeg not found in PATH or pydub cannot access it.")
        print("Please install ffmpeg. On macOS with Homebrew: brew install ffmpeg")
        print("On Debian/Ubuntu: sudo apt-get install ffmpeg")
        return False

def ensure_nltk_punkt_downloaded(current_language_code): # current_language_code not directly used for 'punkt' download itself
    """Checks for NLTK's 'punkt' resource and attempts to download if missing."""
    if not nltk:
        print("Skipping NLTK 'punkt' check: NLTK library not imported/found.")
        return False
    try:
        # Check for the main 'punkt' resource. NLTK's sentence tokenizer
        # will use this to find the appropriate language-specific data.
        nltk.data.find('tokenizers/punkt')
        print("NLTK 'punkt' tokenizer models found.")
        return True
    except LookupError: # This is the correct exception for nltk.data.find failing
        print("NLTK 'punkt' resource not found. Attempting to download...")
        try:
            nltk.download('punkt', quiet=False) # Download the main 'punkt' resource
            print("'punkt' downloaded successfully.")
            # Verify after download
            nltk.data.find('tokenizers/punkt') # Re-check
            print("Verified 'punkt' is now available after download.")
            return True
        except Exception as e_download: # Catch a broader exception for download issues
            print(f"Error during 'punkt' download or verification: {e_download}")
            print("Please try running the following in your Python interpreter and then re-run the script:")
            print("  >>> import nltk")
            print("  >>> nltk.download('punkt')")
            return False
    except AttributeError:
        print("NLTK library seems unavailable. Cannot check/download 'punkt'.")
        return False
    except Exception as e_general:
        print(f"An unexpected error occurred while checking for NLTK 'punkt': {e_general}")
        return False


def create_tts_instance(model_name, device):
    """Initializes and returns the Coqui TTS object."""
    try:
        from TTS.api import TTS as CoquiTTS # Import locally to give clearer error if TTS is missing
        # PyTorch 2.6+ weights_only workaround for XTTS
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
                print("Attempting to apply PyTorch 2.6 workaround: add_safe_globals for XttsConfig")
                torch.serialization.add_safe_globals([XttsConfig])
        except ImportError:
            # This might happen if TTS internal structure changed or XttsConfig is not where expected.
            # It's a warning because the workaround might not be needed for all TTS/PyTorch versions.
            print("WARNING: Could not import XttsConfig (TTS.tts.configs.xtts_config). Workaround may not apply.")
        except Exception as e_sg:
            print(f"WARNING: Error applying add_safe_globals workaround: {e_sg}")

        print(f"Initializing Coqui TTS with model: {model_name} on device: {device}")
        tts_instance = CoquiTTS(model_name=model_name, progress_bar=True)
        tts_instance.to(device)
        print("Coqui TTS initialized successfully.")
        return tts_instance
    except ModuleNotFoundError:
        print("CRITICAL ERROR: The 'TTS' module from Coqui AI was not found.")
        print("Please install it: pip install TTS")
        return None
    except Exception as e:
        print(f"CRITICAL ERROR: Could not initialize Coqui TTS model: {e}")
        # Check for common specific errors to provide more targeted advice
        if "weights_only" in str(e).lower() and "unsupported global" in str(e).lower():
            print("This may be the PyTorch 2.6+ compatibility issue. Ensure PyTorch is an appropriate version (e.g., <2.6 or one known to work with your TTS version) or TTS is updated if a fix exists.")
        elif "'GPT2InferenceModel' object has no attribute 'generate'" in str(e):
            print("This may be an incompatibility with the 'transformers' library version. Try downgrading 'transformers' (e.g., to 4.41.2 or 4.36.2).")
        if device == "mps" and "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
             print("For MPS issues (if you're using 'mps' device), try setting the environment variable: export PYTORCH_ENABLE_MPS_FALLBACK=1")
        return None

def extract_paragraphs_from_docx(docx_path):
    """Extracts and strips paragraphs from a .docx file."""
    try:
        doc = Document(docx_path)
        # Strip each paragraph's text as it's extracted and ensure it's not empty after stripping
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return paragraphs
    except Exception as e:
        print(f"ERROR: Could not read or parse DOCX file '{docx_path}': {e}")
        return []

def clean_text_for_tts(text_input):
    """
    Cleans text for TTS by removing leading/trailing whitespace
    and specific leading patterns like "•\t" or leading tabs/spaces after bullets.
    """
    if not isinstance(text_input, str):
        return "" # Or handle as an error if strictness is required

    text = text_input.strip() # General clean first

    # Order matters: check for more complex patterns first.
    # Example: "• \tText" -> remove "• \t" (3 chars)
    if text.startswith("• \t"):    text = text[3:]
    # Example: "•\tText" -> remove "•\t" (2 chars)
    elif text.startswith("•\t"):  text = text[2:]
    # Example: "• Text" -> remove "• " (2 chars)
    elif text.startswith("• "):    text = text[2:]
    # Example: "•Text" -> remove "•" (1 char)
    elif text.startswith("•"):     text = text[1:]

    # After potentially removing bullet constructs, there might be newly exposed leading whitespace.
    # E.g., if original was "  • \t Text", strip() makes it "• \t Text", pattern removal makes it " Text"
    text = text.lstrip('\t ') # Remove any leading tabs or spaces that are now at the beginning

    return text.strip() # Final overall strip to catch anything else

def split_text_into_chunks(text, language_code, max_length):
    """Splits text into manageable chunks for TTS, first by sentence, then by character limit."""
    if not nltk:
        print("NLTK not available, text will be cleaned but not split by sentence here.")
        cleaned_text = clean_text_for_tts(text)
        return [cleaned_text] if cleaned_text else []

    # 'text' here is a paragraph, assumed to be already stripped by extract_paragraphs_from_docx
    # Use the NLTK language mapping from app_config
    nltk_lang_for_tokenizer = app_config.NLTK_LANGUAGE_MAP.get(language_code.lower(), "english")
    final_chunks = []
    
    try:
        # Ensure 'text' is a string
        if not isinstance(text, str):
            print(f"Warning: Input to split_text_into_chunks is not a string: {type(text)}. Skipping this item.")
            return []

        # Pass the mapped NLTK language name to sent_tokenize
        sentences = nltk.sent_tokenize(text, language=nltk_lang_for_tokenizer)
    except LookupError as e_lookup: # Specifically catch LookupError if language data is missing within punkt
        print(f"NLTK LookupError during sentence tokenization (likely missing language-specific data for '{nltk_lang_for_tokenizer}' within 'punkt'): {e_lookup}")
        print("Ensure 'punkt' is fully downloaded. Processing the paragraph as a single block.")
        sentences = [text] # Fallback
    except Exception as e_tokenize: # Catch other tokenization errors
        print(f"NLTK sentence tokenization failed for paragraph: '{text[:100]}...'. Error: {e_tokenize}.")
        print("Processing the paragraph as a single block for chunking.")
        sentences = [text] # Fallback

    for sentence_text in sentences:
        current_sentence = clean_text_for_tts(sentence_text)
        if not current_sentence:
            continue

        if len(current_sentence) <= max_length:
            final_chunks.append(current_sentence)
        else:
            print(f"  Info: Sentence longer than {max_length} chars, further splitting by words: '{current_sentence[:50]}...'")
            words = current_sentence.split(' ')
            current_chunk = ""
            for word in words:
                if not word: continue
                if len(current_chunk) + len(word) + 1 <= max_length:
                    current_chunk += word + " "
                else:
                    if current_chunk.strip():
                        final_chunks.append(current_chunk.strip())
                    if len(word) > max_length:
                        print(f"    Info: Word longer than {max_length} chars, hard splitting: '{word[:30]}...'")
                        for i in range(0, len(word), max_length):
                            final_chunks.append(word[i:i+max_length])
                        current_chunk = ""
                    else:
                        current_chunk = word + " "
            if current_chunk.strip():
                final_chunks.append(current_chunk.strip())

    return [chunk for chunk in final_chunks if chunk]

def synthesize_text_to_file(tts_instance, text_chunk, speaker_wav, language, output_path):
    """Synthesizes a single text chunk to an audio file."""
    try:
        # The text_chunk is assumed to be fully cleaned by split_text_into_chunks (which uses clean_text_for_tts)
        tts_instance.tts_to_file(
            text=text_chunk,
            speaker_wav=speaker_wav,
            language=language,
            file_path=output_path,
        )
        return True
    except Exception as e:
        print(f"ERROR: Coqui TTS synthesis failed for chunk: '{text_chunk[:60]}...'")
        print(f"  Details: {e}")
        if not os.path.exists(speaker_wav): # Check crucial dependency
            print(f"  CRITICAL ERROR: Speaker WAV file not found at '{speaker_wav}'. This path is from your config or CLI.")
        return False

def combine_audio_files(segment_paths, output_path):
    """Combines multiple WAV files into a single WAV file using pydub."""
    if not segment_paths:
        print("WARNING: No audio segments were successfully synthesized to combine.")
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
    """Main function to orchestrate the DOCX to speech conversion."""
    if not ensure_ffmpeg_is_installed():
        print("ffmpeg issue detected. Audio processing (pydub) might fail or produce poor results.")
        # Depending on severity, you might choose to exit: sys.exit(1)

    # --- Argument Parsing ---
    # Defaults are now pulled from the successfully imported app_config
    parser = argparse.ArgumentParser(description="Convert DOCX file to speech using Coqui XTTS.")
    parser.add_argument("--docx", type=str, default=app_config.DEFAULT_DOCX_PATH,
                        help=f"Path to input .docx file (default: {app_config.DEFAULT_DOCX_PATH})")
    parser.add_argument("--output", type=str, default=app_config.DEFAULT_OUTPUT_FILENAME,
                        help=f"Path to output .wav file (default: {app_config.DEFAULT_OUTPUT_FILENAME})")
    parser.add_argument("--speaker_wav", type=str, default=app_config.SPEAKER_WAV_PATH,
                        help=f"Path to speaker reference .wav (default from config.py: {app_config.SPEAKER_WAV_PATH})")
    parser.add_argument("--language", type=str, default=app_config.LANGUAGE,
                        help=f"Language code for TTS (default from config.py: {app_config.LANGUAGE})")
    parser.add_argument("--device", type=str, default=app_config.DEVICE,
                        help=f"Device for TTS ('cpu','mps','cuda') (default from config.py: {app_config.DEVICE})")
    parser.add_argument("--max_chars", type=int, default=app_config.MAX_CHARS_PER_TTS_SEGMENT,
                        help=f"Max characters per TTS segment (default from config.py: {app_config.MAX_CHARS_PER_TTS_SEGMENT})")
    args = parser.parse_args()

    # Effective settings (config values overridden by CLI arguments if they were provided)
    current_speaker_wav = args.speaker_wav
    current_language = args.language
    current_device = args.device
    docx_file_path = args.docx
    output_file_path = args.output
    max_chars_segment = args.max_chars
    xtts_model_name = app_config.XTTS_MODEL_NAME # This is less likely to be a CLI arg, so take from config

    # NLTK punkt check (uses effective language for mapping, though 'punkt' is general)
    if nltk and not ensure_nltk_punkt_downloaded(current_language):
        print("NLTK 'punkt' resource issue. Sentence splitting capabilities will be affected.")
        # The script will try to continue, but NLTK-based sentence splitting won't work.

    # --- Validations for Critical Paths ---
    if not os.path.exists(current_speaker_wav) or current_speaker_wav == "path/to/your/reference_voice.wav":
        print(f"CRITICAL ERROR: Speaker WAV file not found or is still the placeholder: '{current_speaker_wav}'")
        print("Please update SPEAKER_WAV_PATH in config.py or provide a valid path using the --speaker_wav argument.")
        sys.exit(1)
    if not os.path.exists(docx_file_path):
        print(f"CRITICAL ERROR: DOCX file not found: '{docx_file_path}'")
        sys.exit(1)

    # --- TTS Initialization ---
    tts = create_tts_instance(xtts_model_name, current_device)
    if not tts:
        print("Exiting due to TTS model initialization failure.")
        sys.exit(1)

    # --- Text Processing ---
    paragraphs = extract_paragraphs_from_docx(docx_file_path)
    if not paragraphs:
        print("No processable paragraphs extracted from the DOCX file. Nothing to synthesize.")
        return # Or sys.exit(0) for a clean exit if no work

    # Prepare all chunks for synthesis
    all_text_chunks_for_synthesis = []
    print("Preprocessing text from DOCX and splitting into manageable chunks...")
    for p_idx, paragraph_text in enumerate(paragraphs):
        # Paragraphs are already stripped by extract_paragraphs_from_docx.
        # split_text_into_chunks will apply further cleaning (like bullet removal) to sentences/chunks.
        chunks_from_this_paragraph = split_text_into_chunks(paragraph_text, current_language, max_chars_segment)
        if chunks_from_this_paragraph:
            all_text_chunks_for_synthesis.extend(chunks_from_this_paragraph)
        else:
            print(f"  Info: Paragraph {p_idx+1} (approx '{paragraph_text[:30]}...') resulted in no processable chunks after cleaning/splitting.")

    if not all_text_chunks_for_synthesis:
        print("No text chunks available to synthesize after processing all paragraphs. Exiting.")
        return

    # --- Synthesis Loop ---
    temp_dir = tempfile.mkdtemp(prefix="docx_to_speech_")
    print(f"Using temporary directory for audio segments: {temp_dir}")

    audio_segment_paths = []
    has_synthesis_errors = False
    print(f"\nSynthesizing {len(all_text_chunks_for_synthesis)} text chunks...")

    for i, text_chunk_to_synth in enumerate(tqdm(all_text_chunks_for_synthesis, desc="Synthesizing Chunks")):
        # text_chunk_to_synth is already fully cleaned by the time it gets here
        segment_filename = os.path.join(temp_dir, f"segment_{i:05d}.wav") # Use 5 digits for more chunks
        if synthesize_text_to_file(tts, text_chunk_to_synth, current_speaker_wav, current_language, segment_filename):
            audio_segment_paths.append(segment_filename)
        else:
            has_synthesis_errors = True # Mark that at least one error occurred

    # --- Audio Combination & Cleanup ---
    if not audio_segment_paths: # No segments were successfully created
        print("No audio segments were successfully synthesized. Cannot create a final audio file.")
    else:
        if not combine_audio_files(audio_segment_paths, output_file_path):
            print("Failed to combine audio segments into the final output file.")
            # Decide if errors during combination should also set has_synthesis_errors or a new flag

    try:
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)
        print("Temporary directory cleaned up.")
    except Exception as e:
        print(f"WARNING: Could not clean up temporary directory '{temp_dir}': {e}")
        print("You may need to delete it manually.")

    # --- Final Messages ---
    if has_synthesis_errors:
        print("\nProcess completed, but some errors occurred during speech synthesis for one or more chunks.")
    elif not audio_segment_paths and paragraphs:
        print("\nProcess completed, but no audio could be generated from the provided text (all chunks may have failed or were empty).")
    elif not paragraphs: # Should have been caught, but as a final check
         print("\nProcess completed, but no text was found in the document to process.")
    else: # Implies audio_segment_paths is not empty and no synthesis errors were flagged
        print("\nProcess completed successfully!")

if __name__ == "__main__":
    main()