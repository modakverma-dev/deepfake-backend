
import os
import uvicorn
from fastapi import FastAPI, UploadFile, File
from inference import predict_video, load_model
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="DeepFake Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = None

@app.on_event("startup")
def startup_event():
    global model
    model = load_model()

@app.get("/")
def read_root():
    return {"message": "Model Running..🤖"}

@app.get("/health")
def health_check():
    return {"status": "okay"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    video_bytes = await file.read()
    result = predict_video(video_bytes,model)
    return result

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)