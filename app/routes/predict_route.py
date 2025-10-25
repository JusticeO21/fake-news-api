from fastapi import APIRouter, HTTPException
from app.saved_models.model_loader import get_model
from app.schemas.news_schema import NewsInput
from app.utils.text_cleaner import clean_text
import time

router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.post("/")
async def predict(news_data: NewsInput):
    try:
        start = time.time()
        model, vectorizer = get_model(news_data.model.value)

        cleaned_news = clean_text(news_data.news)
        transformed = vectorizer.transform([cleaned_news])
        if news_data.model.value == "quab":
            transformed = transformed.toarray();
        
        prediction = model.predict(transformed)[0]
        end = time.time()
        probability = model.predict_proba(transformed).max() * 100

        return {"prediction": prediction, "model":news_data.model.value, "qualityScore": f"{probability:.2f}%", "inferenceTime": f"{end - start:.2f}s"}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

