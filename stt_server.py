#!pip install fastapi uvicorn pyngrok openai-whisper torchaudio librosa

from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Optional
import whisper
import torch
import numpy as np
from pyngrok import ngrok
import nest_asyncio
import threading
import uvicorn
import librosa

app = FastAPI(title="Whisper STT API")

# Load Whisper model (adjust based on your needs)
model = whisper.load_model("small")  # Options: tiny, base, small, medium, large

class TranscriptionRequest(BaseModel):
    language: Optional[str] = "en"  # ISO-639-1 language code
    temperature: Optional[float] = 0.0  # Control randomness (0 for deterministic)

@app.post("/transcribe/")
async def transcribe_audio(
    file: UploadFile = File(...),
    request: TranscriptionRequest = None
):
    if not file.content_type.startswith('audio/'):
        raise HTTPException(400, "Only audio files are supported")

    try:
        # Load audio file
        audio_bytes = await file.read()
        audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        
        # Convert to Whisper's expected format
        audio = whisper.pad_or_trim(audio_array)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # Decode options
        options = whisper.DecodingOptions(
            language=request.language if request else "en",
            temperature=request.temperature if request else 0.0,
            fp16=torch.cuda.is_available()
        )

        # Run inference
        result = whisper.decode(model, mel, options)
        
        return {
            "text": result.text,
            "language": options.language,
            "confidence": np.exp(result.no_speech_prob)  # 1 = confident speech
        }

    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {str(e)}")

# Setup Ngrok tunnel
#!ngrok authtoken your api key  # Replace with your actual token
public_url = ngrok.connect(8000)
print(f"Whisper API URL: {public_url}")

# Run FastAPI in background
nest_asyncio.apply()
threading.Thread(
    target=uvicorn.run,
    kwargs={"app": app, "port": 8000, "host": "0.0.0.0"},
    daemon=True
).start()