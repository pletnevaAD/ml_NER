import torch
from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field

from model.consts import DEVICE
from model.model import model
from model.predict import predict_sentence

app = FastAPI(title="NER с BiLSTM+CRF", version="1.0")
model_path = "checkpoints_best/best_model.pth"


@app.on_event("startup")
def startup_event():
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


class TextRequest(BaseModel):
    text: str = Field(description="Текст для распознавания именованных сущностей")


class TokenResponse(BaseModel):
    word: str = Field(description="Токен")
    tag: str = Field(description="NER-тег")


class Prediction(BaseModel):
    text: str = Field(description="Текст для распознавания именованных сущностей")
    tokens: list[TokenResponse] = Field(description="Список токенов с тегами")


@app.post("/predict", response_model=Prediction)
async def predict(request: TextRequest = Depends()):
    words_with_tags = predict_sentence(request.text)
    return Prediction(text=request.text, tokens=words_with_tags)
