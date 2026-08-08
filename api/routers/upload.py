"""Dataset upload endpoint."""
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.schemas import UploadResponse
from src.data.data_loader import DataLoader

router = APIRouter(tags=["upload"])


@router.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".csv", ".xlsx", ".xls"):
        raise HTTPException(status_code=400, detail="Only .csv, .xlsx, .xls files are supported")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        dataset_id = Path(file.filename).stem
        dataset = DataLoader.load(tmp_path, dataset_id=dataset_id)
        return UploadResponse(
            dataset_id=dataset.dataset_id,
            schema=dataset.schema,
            num_rows=len(dataset.df),
            num_columns=len(dataset.df.columns),
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)  # DataLoader already holds the DataFrame in memory
