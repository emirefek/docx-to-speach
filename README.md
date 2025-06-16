Need to put voice sample in voice_samples folder and update config.py

# To Run the Docker Container with GPU Support
docker run --rm --gpus all -v "C:\path\docx-to-speach\data:/data/data" -v "C:\path\docx-to-speach\voice_samples:/data/voice_samples" docx-to-speech-cuda --docx "/data/data/input.docx" --speaker_wav "/data/voice_samples/french.mp3" --output "/data/data/final_audio.wav" --language "tr" --device "cuda"

# To build the container
docker build -t docx-to-speech-cuda .