import os
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document
from src.knowledge.traffic_accident_knowledge import (
    TRAFFIC_ACCIDENT_KNOWLEDGE, 
    TRAFFIC_RULES_KNOWLEDGE, 
    RISK_FACTORS
)

class TrafficAccidentRAGService:
    """
    교통사고 분석을 위한 RAG (Retrieval Augmented Generation) 서비스
    """
    
    def __init__(self, openai_api_key: str = None):
        """
        RAG 서비스 초기화
        
        Args:
            openai_api_key (str): OpenAI API 키
        """
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
            
        self.embeddings = OpenAIEmbeddings(api_key=self.api_key)
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model="gpt-4o-mini",
            temperature=0.1
        )
        self.vectorstore = None
        self.qa_chain = None
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """
        지식 베이스를 벡터 스토어로 초기화
        """
        documents = []
        
        # 교통사고 시나리오 문서화
        for scenario in TRAFFIC_ACCIDENT_KNOWLEDGE:
            content = f"""
            교통사고 시나리오: {scenario['scenario']}
            지표: {', '.join(scenario['indicators'])}
            분석: {scenario['analysis']}
            심각도: {scenario['severity']}
            """
            documents.append(Document(
                page_content=content,
                metadata={"type": "scenario", "scenario": scenario['scenario']}
            ))
        
        # 교통규칙 문서화
        for rule in TRAFFIC_RULES_KNOWLEDGE:
            content = f"""
            교통규칙: {rule['rule']}
            설명: {rule['description']}
            위반 지표: {', '.join(rule['violation_indicators'])}
            """
            documents.append(Document(
                page_content=content,
                metadata={"type": "rule", "rule": rule['rule']}
            ))
        
        # 위험 요소 문서화
        for risk in RISK_FACTORS:
            content = f"""
            위험 요소: {risk['factor']}
            고위험 조건: {', '.join(risk['high_risk'])}
            영향: {risk['impact']}
            """
            documents.append(Document(
                page_content=content,
                metadata={"type": "risk_factor", "factor": risk['factor']}
            ))
        
        # 텍스트 분할 및 벡터스토어 생성
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        splits = text_splitter.split_documents(documents)
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        # QA 체인 생성
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            return_source_documents=True
        )
    
    def analyze_accident_situation(self, detected_objects: List[Dict], context: str = "") -> Dict[str, Any]:
        """
        탐지된 객체들을 바탕으로 교통사고 상황 분석
        
        Args:
            detected_objects (List[Dict]): YOLO로 탐지된 객체들
            context (str): 추가 컨텍스트 정보
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        # 탐지된 객체들을 텍스트로 변환
        objects_text = self._format_detected_objects(detected_objects)
        
        # 쿼리 생성
        query = f"""
        다음과 같은 객체들이 탐지된 교통 상황을 분석해주세요:
        {objects_text}
        
        추가 컨텍스트: {context}
        
        분석해야 할 내용:
        1. 가능한 교통사고 시나리오
        2. 위험도 평가
        3. 위반된 교통규칙 (있다면)
        4. 사고 예방 방법
        5. 사고 원인 분석
        """
        
        # RAG를 통한 분석
        result = self.qa_chain.invoke({"query": query})
        
        return {
            "analysis": result["result"],
            "confidence": self._calculate_confidence(detected_objects),
            "risk_level": self._assess_risk_level(detected_objects),
            "source_documents": [doc.page_content for doc in result["source_documents"]]
        }
    
    def _format_detected_objects(self, detected_objects: List[Dict]) -> str:
        """
        탐지된 객체들을 텍스트 형식으로 포맷
        """
        if not detected_objects:
            return "탐지된 객체가 없습니다."
        
        formatted = []
        for obj in detected_objects:
            formatted.append(
                f"- {obj['object_type']} (신뢰도: {obj['confidence']:.2f})"
            )
        
        return "\n".join(formatted)
    
    def _calculate_confidence(self, detected_objects: List[Dict]) -> float:
        """
        전체적인 분석 신뢰도 계산
        """
        if not detected_objects:
            return 0.0
        
        confidences = [obj['confidence'] for obj in detected_objects]
        return sum(confidences) / len(confidences)
    
    def _assess_risk_level(self, detected_objects: List[Dict]) -> str:
        """
        위험도 평가
        """
        if not detected_objects:
            return "unknown"
        
        object_types = [obj['object_type'] for obj in detected_objects]
        
        # 위험 객체들
        high_risk_objects = ['person', 'bicycle', 'motorcycle']
        medium_risk_objects = ['car', 'truck', 'bus']
        
        if any(obj in high_risk_objects for obj in object_types):
            return "high"
        elif any(obj in medium_risk_objects for obj in object_types):
            return "medium"
        else:
            return "low"
    
    def get_recommendations(self, analysis_result: Dict[str, Any]) -> List[str]:
        """
        분석 결과를 바탕으로 권장사항 생성
        """
        query = f"""
        다음 교통사고 분석 결과를 바탕으로 구체적인 안전 권장사항을 제시해주세요:
        
        분석 결과: {analysis_result['analysis']}
        위험도: {analysis_result['risk_level']}
        
        권장사항은 다음 형태로 제시해주세요:
        1. 즉시 조치사항
        2. 예방 방법
        3. 교통규칙 준수사항
        """
        
        result = self.qa_chain.invoke({"query": query})
        
        # 권장사항을 리스트 형태로 파싱
        recommendations = result["result"].split('\n')
        return [rec.strip() for rec in recommendations if rec.strip()]