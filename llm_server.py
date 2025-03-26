#!pip install -q transformers fastapi uvicorn pyngrok

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pyngrok import ngrok

# Load model with GPU if available
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()

class RequestData(BaseModel):
    prompt: str

@app.post("/generate/")
async def generate_text(data: RequestData):
    try:
        inputs = tokenizer(data.prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_text": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Start ngrok tunnel (replace with your auth token)
#!ngrok authtoken your api token  # Replace with your actual token
public_url = ngrok.connect(8000)
print(f"Public URL: {public_url}")

# Run FastAPI (in background)
import uvicorn
import threading
thread = threading.Thread(target=uvicorn.run, kwargs={"app": app, "port": 8000})
thread.start()