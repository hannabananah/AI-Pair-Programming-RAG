from dotenv import load_dotenv
load_dotenv()

import os
import logging
import time
from typing import List, Optional
from fastapi import FastAPI, Header, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, validator
from openai import OpenAI
from pinecone import Pinecone

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ssafy-knowledge")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4o-mini")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://ai-pair-programming-rag.onrender.com")

# 필수 환경 변수 검증
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY 환경 변수가 설정되지 않았습니다.")

oai = OpenAI(api_key=OPENAI_API_KEY, timeout=30.0)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

app = FastAPI(
    title="SSAFY RAG API",
    version="1.0.0",
    servers=[{"url": PUBLIC_BASE_URL}],
    description="SSAFY 교육생을 위한 AI Pair Programming RAG 서비스",
)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        servers=app.servers,
    )
    for path, methods in openapi_schema.get("paths", {}).items():
        for method, details in methods.items():
            if "requestBody" in details:
                content = details["requestBody"].get("content", {})
                for content_type, schema_info in content.items():
                    if "schema" in schema_info and "properties" in schema_info["schema"]:
                        if "query" in schema_info["schema"]["properties"]:
                            del schema_info["schema"]["properties"]["query"]
                            if "required" in schema_info["schema"]:
                                schema_info["schema"]["required"] = [
                                    r for r in schema_info["schema"]["required"] 
                                    if r != "query"
                                ]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def verify_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    """API 키 검증"""
    if INTERNAL_API_KEY and x_api_key != INTERNAL_API_KEY:
        logger.warning(f"Invalid API key attempt: {x_api_key[:10] if x_api_key else None}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return True

class QueryReq(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="사용자 질문",
        json_schema_extra={"writeOnly": True} 
    )
    top_k: int = Field(default=5, ge=1, le=20, description="검색할 문서 개수")
    namespace: Optional[str] = Field(default="specs", description="Pinecone 네임스페이스")
    include_sources: bool = Field(default=False, description="소스 정보 포함 여부")
    
    @validator('query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError("query는 비어있을 수 없습니다.")
        return v.strip()

class SourceInfo(BaseModel):
    index: int = Field(..., description="소스 순번")
    score: float = Field(..., description="유사도 점수")
    text_preview: Optional[str] = Field(None, description="텍스트 미리보기 (최대 200자)")

class QueryRes(BaseModel):
    answer: str = Field(..., description="AI 응답")
    sources: Optional[List[SourceInfo]] = Field(None, description="참고 소스 정보")
    processing_time: Optional[float] = Field(None, description="처리 시간 (초)")

class ErrorRes(BaseModel):
    error: str = Field(..., description="에러 메시지")
    detail: Optional[str] = Field(None, description="상세 에러 정보")

def retrieve(query: str, namespace: str, top_k: int = 5) -> List[dict]:
    """텍스트만 반환 (파일명/페이지 등 메타데이터 제외)"""
    try:
        logger.info(f"Retrieving documents for query (length: {len(query)}, namespace: {namespace}, top_k: {top_k})")
        q_emb = oai.embeddings.create(
            model=EMBED_MODEL,
            input=[query]
        ).data[0].embedding
        
        res = index.query(
            vector=q_emb,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )
        
        docs = []
        for m in res.matches:
            md = m.metadata or {}
            text = md.get("text", "")
            if not text:
                continue
            docs.append({"score": float(m.score), "text": text})  # 메타데이터 제외, 텍스트만 저장
        
        logger.info(f"Retrieved {len(docs)} documents")
        return docs
    except Exception as e:
        logger.error(f"Error in retrieve: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"문서 검색 중 오류가 발생했습니다: {str(e)}"
        )

def build_prompt(user_q: str, passages: List[dict]) -> list:
    """프롬프트 생성"""
    context = "\n\n".join([f"- {p['text'][:1100]}" for p in passages])  # 문서당 1100자 제한

    sys = (
        "너는 SSAFY 교육생과 페어프로그래밍하는 AI 동료다. 한국어로 간결히 답하고, "
        "정답만 주지 말고 '힌트 → 풀이' 순으로 설명해라. "
        "절대 파일명, 경로, 페이지, 네임스페이스, 인덱스, 내부 스키마, 메타데이터를 언급하지 마라. "
        "참고자료의 원문을 그대로 길게 복사-붙여넣기 하지 말고 요약·재구성해서 설명해라. "
        "확실하지 않으면 모른다고 말하고, 추가 질문이나 방향을 제안해라."
    )
    usr = f"[질문]\n{user_q}\n\n[참고자료 요약용 텍스트]\n{context}"
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]

@app.post(
    "/query",
    response_model=QueryRes,
    responses={
        400: {"model": ErrorRes, "description": "잘못된 요청"},
        401: {"model": ErrorRes, "description": "인증 실패"},
        500: {"model": ErrorRes, "description": "서버 오류"},
    },
    dependencies=[Depends(verify_api_key)],
    summary="RAG 기반 질의응답",
    description="사용자 질문에 대해 벡터 검색을 통해 관련 문서를 찾고, AI가 답변을 생성합니다."
)
async def rag_query(req: QueryReq):
    """RAG 기반 질의응답 엔드포인트"""
    start_time = time.time()
    
    try:
        logger.info(f"Received query request (top_k: {req.top_k}, namespace: {req.namespace})")
        
        passages = retrieve(req.query, req.namespace or "specs", req.top_k)
        
        if not passages:
            logger.warning("No passages retrieved")
            return QueryRes(
                answer="관련된 문서를 찾을 수 없습니다. 질문을 다르게 표현해보시거나 다른 키워드를 사용해보세요.",
                sources=None,
                processing_time=round(time.time() - start_time, 2)
            )
        
        msgs = build_prompt(req.query, passages)
        
        try:
            chat = oai.chat.completions.create(
                model=GEN_MODEL,
                messages=msgs,
                temperature=0.2,
                timeout=30.0
            )
            answer = chat.choices[0].message.content
            
            if not answer:
                raise ValueError("AI 응답이 비어있습니다.")
                
        except Exception as e:
            logger.error(f"Error in OpenAI API call: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"AI 응답 생성 중 오류가 발생했습니다: {str(e)}"
            )
        
        out = {
            "answer": answer,
            "processing_time": round(time.time() - start_time, 2)
        }
        
        if req.include_sources:
            out["sources"] = [
                SourceInfo(
                    index=i+1,
                    score=round(p["score"], 4),
                    text_preview=p["text"][:200] + "..." if len(p["text"]) > 200 else p["text"]
                )
                for i, p in enumerate(passages)
            ]
        
        logger.info(f"Query processed successfully (time: {out['processing_time']}s)")
        return out
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in rag_query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"처리 중 예상치 못한 오류가 발생했습니다: {str(e)}"
        )

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "index_name": INDEX_NAME
    }
