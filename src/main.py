from fastapi import FastAPI
from src.api import endpoints

app = FastAPI(
    title="Traffic Accident Analysis API",
    description="An API to analyze black box videos of traffic accidents.",
    version="1.0.0"
)

# API 라우터 포함
app.include_router(endpoints.router, prefix="/api/v1", tags=["Analysis"])

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the Traffic Accident Analysis API."}

# uvicorn src.main:app --reload
