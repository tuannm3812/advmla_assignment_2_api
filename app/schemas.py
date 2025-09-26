from pydantic import BaseModel, Field
from typing import Dict, List

class VectorPayload(BaseModel):
    features: Dict[str, float] = Field(..., description="feature_name -> value")

class BatchPayload(BaseModel):
    rows: List[Dict[str, float]] = Field(..., description="list of observations") 
