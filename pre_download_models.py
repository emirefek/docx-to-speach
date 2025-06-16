# pre_download_models.py
import os
import torch # Ensure torch is imported for device checks

print("Starting pre-download of Coqui TTS XTTS model...")

# Ensure COQUI_TOS_AGREED is set, as it would be in the Dockerfile environment
os.environ['COQUI_TOS_AGREED'] = '1'

# Determine device (try CUDA, fallback to CPU if not available during build)
# This build step might run on a machine without a GPU (like a CI server)
# or inside a Docker build environment that doesn't have GPU access during the build phase.
build_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Pre-downloading model using device: {build_device}")


# Simplified TTS initialization to trigger download
# We need to handle potential PyTorch 2.6+ issues here too if using such a version
# during build, though the main Dockerfile aims for PyTorch < 2.6.
try:
    from TTS.api import TTS
    
    # --- Minimal PyTorch 2.6+ workaround if necessary during build ---
    # This part is only strictly needed if the PyTorch version during *build* is >= 2.6
    # If your Dockerfile installs PyTorch < 2.6, this is mostly for robustness.
    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import XttsAudioConfig
        if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
            print("Build-time: Applying PyTorch 2.6+ safe_globals workaround for model download.")
            torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig])
    except ImportError:
        print("Build-time: Could not import classes for PyTorch 2.6+ workaround (may not be needed).")
    except Exception as e_sg_build:
        print(f"Build-time: Warning during safe_globals workaround: {e_sg_build}")
    # --- End workaround ---

    # Use the model name from your config.py or hardcode it if simpler for pre-download
    # For simplicity, hardcoding here, ensure it matches your app_config.XTTS_MODEL_NAME
    model_to_download = "tts_models/multilingual/multi-dataset/xtts_v2"
    print(f"Attempting to initialize and download: {model_to_download}")
    
    tts_instance = TTS(model_name=model_to_download, progress_bar=True)
    tts_instance.to(build_device) # Move to device to ensure all parts are loaded/checked

    print(f"Coqui TTS model ({model_to_download}) should now be downloaded to cache.")
    print("Model pre-download script finished successfully.")

except ModuleNotFoundError:
    print("CRITICAL ERROR (Build-time): TTS module not found. Check pip install in Dockerfile.")
    # Propagate error to fail the build
    raise
except Exception as e:
    print(f"CRITICAL ERROR (Build-time): Could not initialize/download Coqui TTS model: {e}")
    # Propagate error to fail the build
    raise