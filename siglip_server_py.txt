import argparse
import base64
import os
from io import BytesIO
import logging

import uvicorn
from fastapi import FastAPI, HTTPException, Body
from transformers import AutoProcessor, AutoModel
import torch
from PIL import Image

parser = argparse.ArgumentParser(description="SigLIP embeddings server")
parser.add_argument("--model_name", default="google/siglip2-so400m-patch16-naflex")
parser.add_argument("--host", default="localhost")
parser.add_argument("--port", type=int, default=5000)
_args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

processor = AutoProcessor.from_pretrained(_args.model_name)
model     = AutoModel.from_pretrained(_args.model_name).eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model   = model.to(device)

app = FastAPI(title="SigLIP Service")

@app.post("/embeddings")
async def embeddings(payload: dict = Body(...)):

    if "content" not in payload:
        raise HTTPException(status_code=400, detail="Missing 'content'")

    inputs = processor(text=payload["content"], return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = model.get_text_features(**inputs)
        vec = out.pooler_output.squeeze(0)
        
    vec = torch.nn.functional.normalize(vec, p=2, dim=0)
    rsp =  vec.cpu().numpy()

    return {"embeddings": rsp.tolist(), "model": _args.model_name}
    
    
@app.post("/embeddings/image")
async def embeddings_image(payload: dict = Body(...)):

    try:
        img_b64 = payload["image_data"][0]["data"]
        img_data = base64.b64decode(img_b64)
        img = Image.open(BytesIO(img_data)).convert("RGB")
        
    except Exception as e:
        log.error(f"Failed to parse image data, {e}")
        raise HTTPException(status_code=400, detail="Invalid image data")    
        
    inputs = processor(images=img, return_tensors="pt").to(device)
        
    with torch.no_grad():
        out = model.get_image_features(**inputs)
        vec = out.pooler_output.squeeze(0)
        
    vec = torch.nn.functional.normalize(vec, p=2, dim=0)
    rsp = vec.cpu().numpy()
    
    return {"embeddings": rsp.tolist(), "model": _args.model_name}
    

if __name__ == "__main__":
    uvicorn.run(app, host=_args.host, port=_args.port)
    
