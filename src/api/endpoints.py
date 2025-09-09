from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from starlette.responses import JSONResponse
from typing import Optional

from src.services.video_processing import VideoProcessingService
from src.services.enhanced_video_service import EnhancedVideoProcessingService
from src.models.domain import AnalysisReport

router = APIRouter()
basic_video_service = VideoProcessingService()
enhanced_video_service = EnhancedVideoProcessingService()

@router.post("/analyze/video", response_model=AnalysisReport)
async def analyze_video_endpoint(
    file: UploadFile = File(...), 
    use_ai: Optional[bool] = Query(default=True, description="AI 기반 고급 분석 사용 여부")
):
    """
    비디오 파일을 업로드하여 교통사고 상황을 분석하고 결과를 반환합니다.

    - **file**: 분석할 비디오 파일 (mp4, avi 등)
    - **use_ai**: AI 기반 고급 분석 사용 여부 (기본값: True)
      - True: ChatGPT + RAG를 활용한 상세 분석
      - False: 기본 YOLO 객체 탐지만 수행
    """
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")

    try:
        if use_ai:
            # AI 기반 고급 분석 사용
            report = await enhanced_video_service.process_video(file)
        else:
            # 기본 YOLO 분석만 사용
            report = await basic_video_service.process_video(file)
        
        return report
    except Exception as e:
        # 실제 프로덕션에서는 더 정교한 예외 처리가 필요합니다.
        return JSONResponse(
            status_code=500,
            content={"message": f"An error occurred during video processing: {e}"}
        )

@router.post("/analyze/video/basic", response_model=AnalysisReport)
async def analyze_video_basic_endpoint(file: UploadFile = File(...)):
    """
    기본 YOLO 객체 탐지만을 수행하는 빠른 분석 엔드포인트

    - **file**: 분석할 비디오 파일 (mp4, avi 등)
    """
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")

    try:
        report = await basic_video_service.process_video(file)
        return report
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"An error occurred during basic video processing: {e}"}
        )

@router.post("/analyze/video/ai", response_model=AnalysisReport)
async def analyze_video_ai_endpoint(file: UploadFile = File(...)):
    """
    AI 기반 고급 분석을 수행하는 엔드포인트 (ChatGPT + RAG)

    - **file**: 분석할 비디오 파일 (mp4, avi 등)
    
    주의: 이 엔드포인트를 사용하려면 OPENAI_API_KEY 환경변수가 설정되어 있어야 합니다.
    """
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")

    try:
        report = await enhanced_video_service.process_video(file)
        return report
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"An error occurred during AI video processing: {e}"}
        )
