#!pip install fastapi uvicorn pyngrok transformers torchaudio coqui-tts

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
import torch
import torchaudio
from TTS.api import TTS
import io
from pyngrok import ngrok
import nest_asyncio
import threading

app = FastAPI()

# Load XTTS-v2 model
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    speaker_wav: Optional[str] = None  # Base64 encoded reference audio

@app.post("/synthesize/")
async def synthesize_speech(request: TTSRequest, file: UploadFile = File(None)):
    try:
        # Use uploaded file or default speaker
        speaker_wav = None
        if file:
            speaker_wav = io.BytesIO(await file.read())
        
        # Generate speech
        wav = tts.tts(
            text=request.text,
            speaker_wav=speaker_wav,
            language=request.language
        )
        
        # Save to buffer
        buffer = io.BytesIO()
        torchaudio.save(buffer, torch.tensor(wav).unsqueeze(0), 24000, format="wav")
        
        return {"audio": buffer.getvalue().hex()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Setup ngrok tunnel
#!ngrok authtoken your api keu
public_url = ngrok.connect(8000)
print(f"XTTS-v2 API URL: {public_url}")

# Run FastAPI in background
nest_asyncio.apply()
threading.Thread(
    target=uvicorn.run,
    kwargs={"app": app, "port": 8000},
    daemon=True
).start()