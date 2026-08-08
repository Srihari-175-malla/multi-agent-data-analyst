"""Pydantic request/response models for the FastAPI app."""
from typing import Dict, List, Optional

from pydantic import BaseModel


class UploadResponse(BaseModel):
    dataset_id: str
    schema: Dict[str, str]
    num_rows: int
    num_columns: int


class AnalyzeRequest(BaseModel):
    dataset_id: str
    question: str


class CriticRound(BaseModel):
    round: int
    verdict: str
    feedback: str


class ToolCall(BaseModel):
    agent: str
    tool: str
    arguments: dict
    result_summary: str
    success: bool
    timestamp: float


class AnalyzeResponse(BaseModel):
    session_id: str
    question: str
    report: str
    chart_paths: List[str]
    critic_rounds: List[CriticRound]
    audit_trail: List[ToolCall]
    revision_rounds_used: int
    elapsed_seconds: float
