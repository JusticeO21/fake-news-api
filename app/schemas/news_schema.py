from pydantic import BaseModel
from enum import Enum

class ModelEnum(str, Enum):
    kbap = "kbap"
    quab = "quab"

class NewsInput(BaseModel):
    news: str
    model: ModelEnum = ModelEnum.kbap

