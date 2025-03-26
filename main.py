import requests
from IPython.display import Audio
import sounddevice as sd
import numpy as np
import whisper
import time

# Configuration
STT_API = "http://your-stt-server/transcribe/"  # Whisper
LLM_API = "http://your-llm-server/generate/"    # DeepSeek
TTS_API = "http://your-tts-server/synthesize/"  # XTTS-v2

def record_audio(duration=5, sr=16000):
    """Record audio from microphone"""
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    return audio.flatten().astype(np.float32)

def stt_to_text(audio_array):
    """Send audio to STT (Whisper)"""
    response = requests.post(
        STT_API,
        files={"file": ("audio.wav", audio_array.tobytes())},
        params={"language": "tr"}  # Turkish for your example
    )
    return response.json()["text"]

def llm_process(text):
    """Get LLM response (DeepSeek)"""
    response = requests.post(
        LLM_API,
        json={"prompt": text, "max_tokens": 150}
    )
    return response.json()["generated_text"]

def tts_to_speech(text):
    """Convert text to speech (XTTS-v2)"""
    response = requests.post(
        TTS_API,
        json={"text": text, "language": "tr"}
    )
    return np.frombuffer(bytes.fromhex(response.json()["audio"]), dtype=np.float32)

def robot_conversation(max_turns=10, silence_timeout=3, stop_phrase="stop"):
    """Humanoid conversation with multiple stop conditions"""
    turn_count = 0
    last_activity_time = time.time()
    
    while True:
        # 1. Stop if max turns reached
        if turn_count >= max_turns:
            print("Maximum conversation turns reached")
            tts_to_speech("Görüşmekten mutluluk duydum! Sonra görüşürüz.")
            break
            
        # 2. Listen with timeout
        try:
            print("\nDinliyorum... (Ctrl+C to stop)")
            audio = record_audio(duration=5)
            
            # 3. Stop if no speech detected for timeout period
            if time.time() - last_activity_time > silence_timeout:
                print(f"No speech detected for {silence_timeout} seconds")
                tts_to_speech("Sesinizi duyamadım. Görüşmeyi sonlandırıyorum.")
                break
                
            # 4. Process speech
            user_text = stt_to_text(audio)
            print(f"You: {user_text}")
            last_activity_time = time.time()
            
            # 5. Stop if user says stop phrase
            if stop_phrase.lower() in user_text.lower():
                tts_to_speech("Görüşme isteğiniz üzerine sonlandırılıyor.")
                print("User requested stop")
                break
                
            # 6. Generate and speak response
            ai_text = llm_process(user_text)
            print(f"AI: {ai_text}")
            tts_to_speech(ai_text)
            turn_count += 1
            
        except KeyboardInterrupt:
            print("\nManual stop detected")
            tts_to_speech("Aniden kesmek zorunda kalıyorum. Hoşçakalın!")
            break
            
        except Exception as e:
            print(f"Error: {e}")
            continue

# Start conversation
robot_conversation()