from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from io import BytesIO

app = FastAPI()

# Serve the static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

# Load pretrained model, tokenizer, and processor
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.post("/generate-captions/")
async def generate_caption(file: UploadFile = File(...)):
    try:
        # Read and preprocess the image
        image = Image.open(BytesIO(await file.read())).convert("RGB")
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)

        # Generate caption
        outputs = model.generate(pixel_values, max_length=30, num_beams=4)
        caption = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return JSONResponse({"caption": caption})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
