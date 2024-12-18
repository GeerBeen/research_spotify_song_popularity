from fastapi import FastAPI, HTTPException
from data_processor import DataProcessor, rf_model
from pydantic import BaseModel, Field, field_validator
from typing_extensions import Annotated
from pydantic import ConfigDict

app = FastAPI()

# Перевірка, чи модель завантажена коректно (для дебагу)
print("RF Model Features:", rf_model.feature_names_in_)


# Модель для запиту
class PredictionRequest(BaseModel):
    duration_ms: Annotated[
        int, Field(ge=1000, le=600000, description="Тривалість у мілісекундах від 1 сек до 10 хвилин")
    ] = 180001
    danceability: Annotated[float, Field(ge=0.0, le=1.0, description="Танцювальність від 0 до 1")] = 0.5
    energy: Annotated[float, Field(ge=0.0, le=1.0, description="Енергійність від 0 до 1")] = 0.5
    loudness: Annotated[float, Field(ge=-60.0, le=0.0, description="Гучність від -60 до 0 dB")] = -10.0
    speechiness: Annotated[float, Field(ge=0.0, le=1.0, description="Частка мови від 0 до 1")] = 0.1
    acousticness: Annotated[float, Field(ge=0.0, le=1.0, description="Акустичність від 0 до 1")] = 0.3
    liveness: Annotated[float, Field(ge=0.0, le=1.0, description="Живість від 0 до 1")] = 0.2
    valence: Annotated[float, Field(ge=0.0, le=1.0, description="Емоційне забарвлення від 0 до 1")] = 0.5
    tempo: Annotated[float, Field(ge=60.0, le=200.0, description="Темп від 60 до 200 BPM")] = 120.0
    instrumentalness: Annotated[float, Field(ge=0.0, le=1.0, description="Інструментальність від 0 до 1")] = 0.0
    track_genre: Annotated[str, Field(min_length=1, description="Назва жанру (непорожній рядок)")] = "pop"

    # Конфігурація Pydantic
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "duration_ms": 180000,
                "danceability": 0.8,
                "energy": 0.7,
                "loudness": -5.0,
                "speechiness": 0.04,
                "acousticness": 0.2,
                "instrumentalness": 0.1,
                "liveness": 0.1,
                "valence": 0.6,
                "tempo": 125.0,
                "track_genre": "pop"
            }
        }
    )

    # Валідація темпу
    @field_validator('tempo')
    @classmethod
    def check_tempo(cls, v):
        if v is not None and (v < 60.0 or v > 200.0):
            raise ValueError("Темп повинен бути у межах від 60 до 200 BPM.")
        return v

    # Валідація жанру
    @field_validator('track_genre')
    @classmethod
    def check_track_genre(cls, v):
        if v is not None and len(v) < 1:
            raise ValueError("Жанр треку має бути непорожнім рядком.")
        return v


# Тестовий маршрут для перевірки API
@app.post("/")
def predict(request: PredictionRequest):
    try:
        # Обробка даних
        data = DataProcessor.preprocess_data(request.model_dump())
        prediction = rf_model.predict(data)
        return {"prediction": float(prediction[0])}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Неочікувана помилка: {str(e)}")
