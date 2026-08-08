"""Analysis endpoint — runs the full multi-agent workflow."""
from fastapi import APIRouter, HTTPException

from api.schemas import AnalyzeRequest, AnalyzeResponse
from src.orchestration.workflow import AnalysisWorkflow

router = APIRouter(tags=["analyze"])
_workflow = None


def get_workflow() -> AnalysisWorkflow:
    global _workflow
    if _workflow is None:
        _workflow = AnalysisWorkflow()
    return _workflow


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    try:
        result = get_workflow().run(request.dataset_id, request.question)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return result.__dict__
