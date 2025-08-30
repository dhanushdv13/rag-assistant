from pydantic import BaseModel
from typing import List, Optional

class AskRequest(BaseModel):
    question: str

class Source(BaseModel):
    source: Optional[str] = None
    page: Optional[int] = None
    snippet: str

class AskResponse(BaseModel):
    answer: str
    sources: List[Source] = []

class HealthResponse(BaseModel):
    status: str
    vectorstore: str
    qa_chain: str
    custom_model: str
    device: str
    provider: str

class ModelInfo(BaseModel):
    name: str
    description: str
    type: str
    status: str
    path: Optional[str] = None

class AvailableModelsResponse(BaseModel):
    available_models: dict
    current_model: str
    current_provider: str
    custom_model_exists: bool
