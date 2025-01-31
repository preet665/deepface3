from fastapi import FastAPI, File, UploadFile
from deepface import DeepFace
from pydantic import BaseModel
from PIL import Image
import io
import uvicorn

class EmotionAnalysisResponse(BaseModel):
    dominant_emotion: str
    emotion_scores: dict
    face_confidence: float

app = FastAPI()

@app.post("/analyze-emotion/")
async def analyze_emotion(file: UploadFile = File(...)):
    try:
        # Read image file
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Save temporarily for DeepFace to process
        temp_path = "temp_image.jpg"
        image.save(temp_path)

        # Analyze emotion
        result = DeepFace.analyze(temp_path, actions=['emotion'])
        emotion_data = result[0]
        dominant_emotion = emotion_data['dominant_emotion']
        emotion_scores = emotion_data['emotion']
        
        emotion_scores = {key: float(value) for key, value in emotion_scores.items()}
        face_confidence = float(emotion_data['face_confidence'])
        
        # Return the data as an object (Pydantic model)
        return EmotionAnalysisResponse(
            dominant_emotion=dominant_emotion,
            emotion_scores=emotion_scores,
            face_confidence=face_confidence
        )

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    print('abc')
    uvicorn.run(app, host="0.0.0.0", port=8000)
