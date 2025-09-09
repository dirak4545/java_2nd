import tempfile
import os
import cv2
from fastapi import UploadFile
from typing import List, Dict, Any

from src.analysis.video_analyzer import VideoAnalyzer
from src.services.rag_service import TrafficAccidentRAGService
from src.models.domain import AnalysisReport, FrameAnalysis

class EnhancedVideoProcessingService:
    """
    AI 고도화된 비디오 처리 서비스 (YOLO + ChatGPT + RAG)
    """
    
    def __init__(self):
        self.analyzer = VideoAnalyzer()
        try:
            self.rag_service = TrafficAccidentRAGService()
        except ValueError as e:
            print(f"Warning: RAG service initialization failed: {e}")
            self.rag_service = None
    
    async def process_video(self, video_file: UploadFile) -> AnalysisReport:
        """
        비디오를 처리하고 AI 기반 상세 분석 수행
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.filename)[1]) as tmp_file:
            content = await video_file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # 1. YOLO를 통한 객체 탐지
            frame_analysis_results = self.analyzer.analyze(tmp_file_path)
            
            # 2. 비디오 메타데이터 추출
            video_metadata = self._extract_video_metadata(tmp_file_path)
            
            # 3. AI 기반 종합 분석
            ai_analysis = await self._perform_ai_analysis(frame_analysis_results, video_metadata)
            
            # 4. 최종 리포트 생성
            report = AnalysisReport(
                filename=video_file.filename,
                duration=video_metadata['duration'],
                summary=ai_analysis['summary'],
                frames=frame_analysis_results,
                ai_analysis=ai_analysis,  # 새로운 필드 추가
                risk_assessment=ai_analysis.get('risk_assessment', {}),
                recommendations=ai_analysis.get('recommendations', [])
            )
            
            return report
            
        finally:
            os.unlink(tmp_file_path)
    
    def _extract_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """
        비디오 메타데이터 추출
        """
        cap = cv2.VideoCapture(video_path)
        
        # 기본 정보 추출
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': duration,
            'resolution': f"{width}x{height}"
        }
    
    async def _perform_ai_analysis(self, frames: List[FrameAnalysis], metadata: Dict) -> Dict[str, Any]:
        """
        AI 기반 종합 상황 분석
        """
        if not self.rag_service:
            return {
                'summary': '기본적인 객체 탐지가 완료되었습니다. AI 분석을 위해서는 OpenAI API 키가 필요합니다.',
                'risk_assessment': {'level': 'unknown', 'confidence': 0.0},
                'recommendations': ['OpenAI API 키를 설정하여 고급 분석을 사용하세요.']
            }
        
        # 모든 프레임에서 탐지된 객체들 수집
        all_detected_objects = []
        for frame in frames:
            for obj in frame.detected_objects:
                all_detected_objects.append({
                    'object_type': obj.object_type,
                    'confidence': obj.confidence,
                    'timestamp': frame.timestamp
                })
        
        # 컨텍스트 정보 생성
        context = self._generate_context(all_detected_objects, metadata)
        
        # RAG 기반 상황 분석
        try:
            analysis_result = self.rag_service.analyze_accident_situation(
                detected_objects=all_detected_objects,
                context=context
            )
            
            # 권장사항 생성
            recommendations = self.rag_service.get_recommendations(analysis_result)
            
            # 상세 위험도 평가
            risk_assessment = self._detailed_risk_assessment(all_detected_objects, analysis_result)
            
            # 종합 요약 생성
            summary = self._generate_comprehensive_summary(
                analysis_result, risk_assessment, metadata
            )
            
            return {
                'summary': summary,
                'detailed_analysis': analysis_result['analysis'],
                'risk_assessment': risk_assessment,
                'recommendations': recommendations,
                'ai_confidence': analysis_result['confidence']
            }
            
        except Exception as e:
            print(f"AI analysis error: {e}")
            return {
                'summary': f'객체 탐지는 완료되었으나 AI 분석 중 오류가 발생했습니다: {str(e)}',
                'risk_assessment': {'level': 'unknown', 'confidence': 0.0},
                'recommendations': ['시스템 설정을 확인하고 다시 시도해주세요.']
            }
    
    def _generate_context(self, detected_objects: List[Dict], metadata: Dict) -> str:
        """
        분석을 위한 컨텍스트 정보 생성
        """
        # 객체 통계
        object_counts = {}
        for obj in detected_objects:
            obj_type = obj['object_type']
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
        
        # 시간대별 객체 분포
        timeline_analysis = self._analyze_timeline(detected_objects)
        
        context = f"""
        비디오 정보:
        - 길이: {metadata['duration']:.1f}초
        - 해상도: {metadata['resolution']}
        - FPS: {metadata['fps']:.1f}
        
        탐지된 객체 통계:
        {self._format_object_counts(object_counts)}
        
        시간대별 분석:
        {timeline_analysis}
        """
        
        return context
    
    def _analyze_timeline(self, detected_objects: List[Dict]) -> str:
        """
        시간대별 객체 분포 분석
        """
        if not detected_objects:
            return "탐지된 객체가 없습니다."
        
        # 시간대별 그룹화 (1초 단위)
        timeline_groups = {}
        for obj in detected_objects:
            time_group = int(obj['timestamp'])
            if time_group not in timeline_groups:
                timeline_groups[time_group] = []
            timeline_groups[time_group].append(obj['object_type'])
        
        timeline_summary = []
        for time_group in sorted(timeline_groups.keys()):
            objects = timeline_groups[time_group]
            unique_objects = list(set(objects))
            timeline_summary.append(f"{time_group}초: {', '.join(unique_objects)}")
        
        return '\n'.join(timeline_summary[:10])  # 최대 10개 시간대만 표시
    
    def _format_object_counts(self, object_counts: Dict[str, int]) -> str:
        """
        객체 개수를 포맷팅
        """
        if not object_counts:
            return "- 탐지된 객체 없음"
        
        formatted = []
        for obj_type, count in object_counts.items():
            formatted.append(f"- {obj_type}: {count}회 탐지")
        
        return '\n'.join(formatted)
    
    def _detailed_risk_assessment(self, detected_objects: List[Dict], analysis_result: Dict) -> Dict[str, Any]:
        """
        상세 위험도 평가
        """
        base_risk = analysis_result.get('risk_level', 'low')
        confidence = analysis_result.get('confidence', 0.0)
        
        # 객체별 위험도 가중치
        risk_weights = {
            'person': 1.0,      # 보행자 최고 위험
            'bicycle': 0.8,     # 자전거
            'motorcycle': 0.8,  # 오토바이
            'car': 0.6,         # 자동차
            'truck': 0.7,       # 트럭
            'bus': 0.7          # 버스
        }
        
        # 가중 평균 위험도 계산
        total_weight = 0
        total_risk = 0
        
        for obj in detected_objects:
            obj_type = obj['object_type']
            if obj_type in risk_weights:
                weight = risk_weights[obj_type] * obj['confidence']
                total_weight += weight
                total_risk += weight
        
        calculated_risk = total_risk / total_weight if total_weight > 0 else 0
        
        # 위험도 레벨 결정
        if calculated_risk > 0.8:
            risk_level = 'very_high'
        elif calculated_risk > 0.6:
            risk_level = 'high'
        elif calculated_risk > 0.4:
            risk_level = 'medium'
        elif calculated_risk > 0.2:
            risk_level = 'low'
        else:
            risk_level = 'very_low'
        
        return {
            'level': risk_level,
            'score': calculated_risk,
            'confidence': confidence,
            'factors': self._identify_risk_factors(detected_objects)
        }
    
    def _identify_risk_factors(self, detected_objects: List[Dict]) -> List[str]:
        """
        위험 요소 식별
        """
        factors = []
        object_types = [obj['object_type'] for obj in detected_objects]
        
        if 'person' in object_types:
            factors.append('보행자 존재')
        if any(vehicle in object_types for vehicle in ['car', 'truck', 'bus']):
            factors.append('차량 운행')
        if 'bicycle' in object_types or 'motorcycle' in object_types:
            factors.append('이륜차 운행')
        
        # 밀도 분석
        if len(detected_objects) > 50:
            factors.append('높은 교통 밀도')
        elif len(detected_objects) > 20:
            factors.append('중간 교통 밀도')
        
        return factors
    
    def _generate_comprehensive_summary(self, analysis_result: Dict, risk_assessment: Dict, metadata: Dict) -> str:
        """
        종합적인 요약 생성
        """
        duration = metadata['duration']
        risk_level = risk_assessment['level']
        
        summary = f"""
        {duration:.1f}초 길이의 교통 영상 분석이 완료되었습니다.
        
        위험도 평가: {risk_level.upper()} (신뢰도: {risk_assessment['confidence']:.1%})
        
        주요 발견사항:
        {analysis_result.get('analysis', '상세 분석을 확인하세요.')}
        
        식별된 위험 요소: {', '.join(risk_assessment.get('factors', ['없음']))}
        """
        
        return summary.strip()