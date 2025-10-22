from fastapi import APIRouter, HTTPException
from app.saved_models.model_loader import get_model
from app.schemas.news_schema import NewsInput
from app.utils.text_cleaner import clean_text

router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.post("/")
async def predict(news_data: NewsInput):
    try:
        model, vectorizer = get_model(news_data.model.value)

        cleaned_news = clean_text(news_data.news)
        transformed = vectorizer.transform([cleaned_news])
        prediction = model.predict(transformed)[0]

        return {"prediction": prediction}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
