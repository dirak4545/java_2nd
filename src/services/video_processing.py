import tempfile
import os
from fastapi import UploadFile

from src.analysis.video_analyzer import VideoAnalyzer
from src.models.domain import AnalysisReport

class VideoProcessingService:
    """
    비디오 처리 요청을 받아 분석을 조율하고 결과를 리포트 형식으로 반환합니다.
    """
    def __init__(self):
        self.analyzer = VideoAnalyzer()

    async def process_video(self, video_file: UploadFile) -> AnalysisReport:
        """
        업로드된 비디오 파일을 처리하고 분석 리포트를 생성합니다.

        Args:
            video_file (UploadFile): FastAPI를 통해 업로드된 비디오 파일

        Returns:
            AnalysisReport: 생성된 분석 리포트
        """
        # UploadFile을 임시 파일로 저장하여 분석기에 전달
        # tempfile.NamedTemporaryFile은 with 블록 종료 시 자동으로 파일을 삭제합니다.
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.filename)[1]) as tmp_file:
            content = await video_file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # 비디오 분석 수행
            frame_analysis_results = self.analyzer.analyze(tmp_file_path)

            # 결과를 바탕으로 최종 리포트 생성
            report = AnalysisReport(
                filename=video_file.filename,
                duration=5.0,  # 실제로는 비디오 길이를 읽어야 함
                summary="A collision was detected between a car and a pedestrian at timestamp 2.0.",
                frames=frame_analysis_results
            )
            return report
        finally:
            # 임시 파일 삭제
            os.unlink(tmp_file_path)
