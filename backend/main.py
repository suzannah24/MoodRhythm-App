import joblib
import numpy as np
import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enhanced CORS for production deployment stability
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows your Vercel frontend to communicate securely
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FaceData(BaseModel):
    smile_w: float; smile_h: float; brow_h: float; eye_h: float; brow_i: float
    mouth_drop: float; brow_v_left: float; brow_v_right: float; lip_dist: float; jaw_dist: float

print("--- MOODRHYTHM BACKEND INITIALIZING ---")
try:
    model = joblib.load("mood_classifier.pkl")
    scaler = joblib.load("scaler.pkl")
    print("✅ SUCCESS: AI Brain and Scaler loaded.")
except Exception as e:
    print(f"❌ ERROR DURING STARTUP: {e}")

@app.post("/analyze-face")
async def analyze_face(data: FaceData):
    try:
        raw_features = np.array([[
            data.smile_w, data.smile_h, data.brow_h, data.eye_h, data.brow_i,
            data.mouth_drop, data.brow_v_left, data.brow_v_right, data.lip_dist, data.jaw_dist
        ]])
        scaled_features = scaler.transform(raw_features)
        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]
        confidence = float(np.max(probabilities))

        print(f"Detected: {prediction.upper()} ({confidence:.2%})")
        return {"detected_mood": prediction, "confidence": confidence}
    except Exception as e:
        print(f"Prediction Error: {e}")
        return {"error": str(e)}

@app.post("/save-feedback")
async def save_feedback(data: FaceData, confirmed_mood: str):
    file_path = "user_feedback_data.csv"
    file_exists = os.path.isfile(file_path)
    try:
        with open(file_path, "a") as f:
            if not file_exists:
                f.write("label,smile_w,smile_h,brow_h,eye_h,brow_i,mouth_drop,brow_v_left,brow_v_right,lip_dist,jaw_dist\n")
            
            row = f"{confirmed_mood},{data.smile_w},{data.smile_h},{data.brow_h},{data.eye_h},{data.brow_i}," \
                  f"{data.mouth_drop},{data.brow_v_left},{data.brow_v_right},{data.lip_dist},{data.jaw_dist}\n"
            f.write(row)
            
        print(f"💾 TARGETED FEEDBACK LOGGED: Saved as {confirmed_mood.upper()}")
        return {"status": "success"}
    except Exception as e:
        print(f"Feedback Error: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)