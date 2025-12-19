import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from detector import AutonomousDecisionSystem

app = FastAPI()

# Enhanced CORS for portability across different laptop IPs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use /tmp for cloud deployment (Vercel) but keep local folders for laptops
# This detects if it's running in a cloud environment
is_cloud = "VERCEL" in os.environ
base_dir = "/tmp" if is_cloud else "."

UPLOAD_DIR = os.path.join(base_dir, "uploads")
OUTPUT_DIR = os.path.join(base_dir, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount the output directory so processed images/videos can be viewed
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# Initialize the system. Ensure best.pt is in the root folder
system = AutonomousDecisionSystem(model_path="best.pt")

processing_status = {
    "is_processing": False, 
    "progress": 0, 
    "total_frames": 0, 
    "current_frame": 0
}

# --- FRONTEND ROUTES ---
# These allow Python to serve your HTML directly
@app.get("/")
async def serve_landing():
    # Looks for landing.html in the same folder as main.py
    return FileResponse("landing.html")

@app.get("/index.html")
async def serve_app():
    # Looks for index.html in the same folder as main.py
    return FileResponse("index.html")

# --- API ENDPOINTS ---

@app.post("/process-media/")
async def process_media_endpoint(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower()
    is_video = ext in ['.mp4', '.avi', '.mov', '.mkv']
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if is_video:
            output_filename = f"processed_{file_id}.mp4"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            processing_status.update({"is_processing": True, "progress": 0, "current_frame": 0})
            
            # Process the video using the detector logic
            final_decision = system.process_video(input_path, output_path, processing_status)
            
            processing_status.update({"is_processing": False, "progress": 100})
            media_type = "video"
        else:
            output_filename = f"processed_{file_id}.jpg"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            final_decision = system.process_image(input_path, output_path)
            media_type = "image"
        
        return {
            "status": "success",
            "media_url": f"/outputs/{output_filename}",
            "final_decision": final_decision,
            "media_type": media_type
        }
    except Exception as e:
        processing_status["is_processing"] = False
        return {"status": "error", "message": str(e)}

@app.get("/processing-progress")
def get_processing_progress():
    return processing_status

@app.get("/video_feed")
def video_feed():
    # Live webcam feed logic from detector.py
    return StreamingResponse(system.generate_frames(), 
                             media_type='multipart/x-mixed-replace; boundary=frame')

@app.post("/stop-feed-signal")
def stop_feed_signal():
    system.stop_streaming()
    return {"status": "stopped"}

@app.get("/current-status")
def get_current_status():
    return {"action": system.latest_action}

if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0 makes the server accessible to other laptops on the same Wi-Fi
    print("🚀 Server Ready. Go to http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
