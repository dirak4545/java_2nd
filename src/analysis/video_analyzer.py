from typing import List
import torch
from ultralytics import YOLO

from src.models.domain import FrameAnalysis, DetectedObject

class VideoAnalyzer:
    """
    YOLOv8 모델을 사용하여 비디오를 분석하고 객체 탐지를 수행합니다.
    """
    def __init__(self):
        """
        VideoAnalyzer를 초기화하고 YOLOv8 모델을 로드합니다.
        GPU 사용 가능 여부를 확인하고, 가능하면 GPU를 사용합니다.
        """
        # GPU 장치 확인 (NVIDIA CUDA 또는 Apple Silicon MPS)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"Using device: {self.device}")

        # YOLOv8 모델 로드 (가장 작고 빠른 'nano' 모델)
        # 처음 실행 시 모델 가중치를 다운로드합니다.
        self.model = YOLO('yolov8n.pt')
        self.model.to(self.device)

    def analyze(self, video_path: str) -> List[FrameAnalysis]:
        """
        주어진 비디오 경로에서 파일을 읽고 YOLOv8 모델로 분석을 수행합니다.

        Args:
            video_path (str): 분석할 비디오 파일의 경로

        Returns:
            List[FrameAnalysis]: 프레임별 분석 결과 리스트
        """
        print(f"Analyzing video with YOLOv8: {video_path}")
        
        # stream=True는 비디오 같은 긴 소스에 대해 메모리 효율적인 처리를 가능하게 합니다.
        results = self.model.predict(source=video_path, stream=True, device=self.device)
        
        analysis_frames: List[FrameAnalysis] = []
        
        # 모델이 인식하는 클래스 이름 가져오기
        class_names = self.model.names

        frame_timestamp = 0.0
        for i, result in enumerate(results):
            # YOLO 결과에서 탐지된 객체 정보 추출
            detected_objects: List[DetectedObject] = []
            
            # `result.boxes`에서 Bounding Box, Confidence, Class ID 정보 가져오기
            for box in result.boxes:
                try:
                    xyxy = box.xyxy[0].tolist()
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    class_name = class_names.get(class_id, "unknown")

                    detected_obj = DetectedObject(
                        object_type=class_name,
                        confidence=confidence,
                        bounding_box=tuple(map(int, xyxy))
                    )
                    detected_objects.append(detected_obj)
                except (IndexError, KeyError) as e:
                    print(f"Could not process a detection due to an error: {e}")

            # 현재 프레임의 분석 결과 생성
            # 참고: 실제 timestamp를 얻으려면 비디오의 FPS 정보가 필요합니다.
            # 여기서는 프레임 인덱스를 기반으로 간단히 가정합니다. (30fps 가정)
            frame_timestamp = i / 30.0 
            
            frame_analysis = FrameAnalysis(
                timestamp=frame_timestamp,
                detected_objects=detected_objects
            )
            analysis_frames.append(frame_analysis)

        print("Analysis complete.")
        return analysis_frames
