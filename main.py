from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
import shutil
import uuid
import os
from gpt1_pointillist_stego import encode_text_into_image

app = FastAPI()

@app.post("/encode")
async def encode(text: str = Form(...), image: UploadFile = Form(...)):
    # Generate unique filenames
    input_path = f"temp_{uuid.uuid4().hex}.png"
    output_path = f"encoded_{uuid.uuid4().hex}.png"
    
    try:
        # Save uploaded image
        with open(input_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
        
        # Run encoding
        try:
            encode_text_into_image(text, input_path, output_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail="Internal server error during encoding")
        
        # Return the encoded image
        return FileResponse(
            output_path,
            media_type="image/png",
            filename="encoded.png",
            background=None  # This ensures the file is deleted after sending
        )
    
    except Exception as e:
        # Clean up any temporary files in case of error
        for path in [input_path, output_path]:
            if os.path.exists(path):
                os.remove(path)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up input file
        if os.path.exists(input_path):
            os.remove(input_path)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")