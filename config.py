# config.py

# --- XTTS Configuration ---
# REQUIRED: Update this path to your actual reference speaker audio file
SPEAKER_WAV_PATH = "/data/voice_samples/female.mp3"

# Default language for TTS and sentence splitting (e.g., "tr", "en", "es")
LANGUAGE = "tr"

# Coqui XTTS model name
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# Device for TTS processing ("cpu", "mps", "cuda")
# For Apple Silicon (M-series Macs), "mps" can be used.
# If MPS causes issues, "cpu" is a fallback.
# Set PYTORCH_ENABLE_MPS_FALLBACK=1 in your environment if using "mps" and encountering errors.
DEVICE = "cuda"

# --- Application Settings ---
# Default input DOCX file if no CLI argument is provided
DEFAULT_DOCX_PATH = "/data/input.docx" # Or some other sensible default

# Default output audio filename
DEFAULT_OUTPUT_FILENAME = "/data/output.wav"

# Maximum characters per single text segment sent to the TTS engine.
# This helps manage potential truncation or quality issues with very long segments,
# especially for certain languages (e.g., Turkish 'tr' had a warning around 226).
# Adjust based on observed behavior and model/language limitations.
MAX_CHARS_PER_TTS_SEGMENT = 220

# --- NLTK Settings ---
# Mapping of language codes to NLTK's sentence tokenizer language names
NLTK_LANGUAGE_MAP = {
    "en": "english",
    "tr": "turkish",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "pt": "portuguese"
    # Add more languages as needed for NLTK sentence tokenization
}

# --- Temporary File Settings ---
# You generally don't need to change this, as the script uses a system temp directory.
# TEMP_DIR_NAME = ".temp_audio_segments_xtts" # (If you preferred a local temp dir)

# --- PyTorch 2.6 Workaround ---
# This is handled in the main script's create_tts_instance function.
# No direct config variable here, but noting its relevance.