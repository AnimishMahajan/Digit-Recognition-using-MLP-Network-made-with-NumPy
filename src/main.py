import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os
import sys

import nn_npy as nn

app = FastAPI()

# Setup templates
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)

# Initialize Network and Load Model
model = nn.network()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mnist")

print(f"Loading model from: {MODEL_PATH}")
if os.path.exists(os.path.join(MODEL_PATH, "params.npz")):
    model.load_model(MODEL_PATH)
    print("Model loaded successfully.")
else:
    print(f"WARNING: Model not found at {MODEL_PATH}. Prediction will fail or use random weights.")

class ImagePayload(BaseModel):
    image: str

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_digit(payload: ImagePayload):
    try:
        # Parse base64 image
        # Format: "data:image/png;base64,....."
        if "," in payload.image:
            header, encoded = payload.image.split(",", 1)
        else:
            encoded = payload.image
            
        image_data = base64.b64decode(encoded)
        image = Image.open(BytesIO(image_data))
        
        # Ensure image is resized to 28x28 and grayscale
        # L = 8-bit pixels, black and white
        image = image.convert("L").resize((28, 28))
        
        # Convert to numpy array
        # User specified: "Image background must be black and drawing must be white"
        # We assume the frontend sends it exactly like that.
        img_array = np.array(image)
        
        # Flatten/Reshape to (-1, 1) as requested
        # Also normally models expect normalized data 0-1
        img_array = img_array.astype(np.float32) / 255.0
        
        input_data = (img_array.reshape(-1, 1))
        
        # Predict
        # predict returns the output layer activations
        output = model.predict(input_data)
        
        # Determine predicted class
        # output is a numpy array of shape (10, 1) likely
        predicted_number = int(np.argmax(output))

        return JSONResponse({
            "prediction": predicted_number,
            "raw_output": output.tolist()
        })
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
