from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse
import shutil
import uuid
from gpt1_pointillist_stego import encode_text_into_image  # adjust if different

app = FastAPI()

@app.post("/encode")
async def encode(text: str = Form(...), image: UploadFile = Form(...)):
    # Save uploaded image
    input_path = f"temp_{uuid.uuid4().hex}.png"
    with open(input_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    # Generate output path
    output_path = f"encoded_{uuid.uuid4().hex}.png"

    # Run encoding
    encode_text_into_image(text, input_path, output_path)

    return FileResponse(output_path, media_type="image/png", filename="encoded.png")