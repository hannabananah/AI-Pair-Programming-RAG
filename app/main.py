from dotenv import load_dotenv
load_dotenv()

import os
from typing import List, Optional
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = os.getenv("PINECONE_INDEX_NAME", "ssafy-knowledge")
EMBED_MODEL      = os.getenv("EMBED_MODEL", "text-embedding-3-small")
GEN_MODEL        = os.getenv("GEN_MODEL", "gpt-4o-mini")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")
PUBLIC_BASE_URL  = os.getenv("PUBLIC_BASE_URL", "https://ai-pair-programming-rag.onrender.com")

oai = OpenAI(api_key=OPENAI_API_KEY)
pc  = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

app = FastAPI(
    title="SSAFY RAG API",
    version="1.0.0",
    servers=[{"url": PUBLIC_BASE_URL}],
)

def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    if INTERNAL_API_KEY and x_api_key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

class QueryReq(BaseModel):
    query: str
    top_k: int = 5
    namespace: Optional[str] = "specs"
    include_sources: bool = False

class QueryRes(BaseModel):
    answer: str
    sources: Optional[List[dict]] = None

def retrieve(query: str, namespace: str, top_k: int = 5):
    """Pinecone에서 텍스트만 가져온다(파일명/페이지 등 메타 정보 제외)."""
    q_emb = oai.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    res = index.query(vector=q_emb, top_k=top_k, include_metadata=True, namespace=namespace)
    docs = []
    for m in res.matches:
        md = m.metadata or {}
        text = md.get("text", "")
        if not text:
            continue
        docs.append({"score": float(m.score), "text": text})
    return docs

def build_prompt(user_q: str, passages: List[dict]) -> list:
    # 컨텍스트는 텍스트만, 어떤 식별자/출처도 포함 금지
    context = "\n\n".join([f"- {p['text'][:1100]}" for p in passages])

    sys = (
        "너는 SSAFY 교육생과 페어프로그래밍하는 AI 동료다. 한국어로 간결히 답하고, "
        "정답만 주지 말고 '힌트 → 풀이' 순으로 설명해라. "
        "절대 파일명, 경로, 페이지, 네임스페이스, 인덱스, 내부 스키마, 메타데이터를 언급하지 마라. "
        "참고자료의 원문을 그대로 길게 복사-붙여넣기 하지 말고 요약·재구성해서 설명해라. "
        "확실하지 않으면 모른다고 말하고, 추가 질문이나 방향을 제안해라."
    )
    usr = f"[질문]\n{user_q}\n\n[참고자료 요약용 텍스트]\n{context}"
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]

@app.post("/query", response_model=QueryRes, dependencies=[Depends(verify_api_key)])
def rag_query(req: QueryReq):
    passages = retrieve(req.query, req.namespace or "specs", req.top_k)
    msgs = build_prompt(req.query, passages)

    chat = oai.chat.completions.create(
        model=GEN_MODEL,
        messages=msgs,
        temperature=0.2
    )
    answer = chat.choices[0].message.content

    out = {"answer": answer}

    if req.include_sources:
        out["sources"] = [{"i": i+1, "score": p["score"]} for i, p in enumerate(passages)]

    return out

from fastapi.responses import RedirectResponse
@app.get("/")
def root():
    return RedirectResponse(url="/docs")
