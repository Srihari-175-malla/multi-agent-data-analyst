"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import analyze, upload

app = FastAPI(
    title="Multi-Agent AI Data Analyst API",
    description="Upload tabular data and get evidence-backed answers from a team of SQL/Python/Statistics agents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(analyze.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
