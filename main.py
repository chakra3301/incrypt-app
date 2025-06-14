from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
import shutil
import uuid
import os
from gpt1_pointillist_stego import encode_text_into_image

app = FastAPI(
    title="Steganography API",
    description="API for encoding text into images using steganography",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/encode", 
    response_description="Returns the encoded image file",
    responses={
        200: {
            "description": "Successfully encoded image",
            "content": {
                "image/png": {
                    "example": "Binary image data"
                }
            }
        },
        400: {
            "description": "Bad request - text too long or invalid input",
            "content": {
                "application/json": {
                    "example": {"detail": "Text too long. Maximum capacity is 1000 bytes, but text is 2000 bytes"}
                }
            }
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {"detail": "Internal server error during encoding"}
                }
            }
        }
    }
)
async def encode(text: str = Form(..., description="The text to encode into the image"),
                image: UploadFile = Form(..., description="The image file to encode the text into")):
    """
    Encode text into an image using steganography.
    
    - **text**: The text you want to encode into the image
    - **image**: The image file (PNG format recommended) to encode the text into
    
    Returns the encoded image file.
    """
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
    """Redirect to API documentation"""
    return RedirectResponse(url="/docs")

# Health check endpoint
@app.get("/health")
def health_check():
    """Check if the API is running"""
    return {"status": "healthy"}