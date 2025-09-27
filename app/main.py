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
    servers=[{"url": PUBLIC_BASE_URL}]
)

def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    if INTERNAL_API_KEY and x_api_key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

class QueryReq(BaseModel):
    query: str
    top_k: int = 5
    namespace: Optional[str] = "specs"
    include_sources: bool = True

class QueryRes(BaseModel):
    answer: str
    sources: Optional[List[dict]] = None

def retrieve(query: str, namespace: str, top_k: int = 5):
    q_emb = oai.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    res = index.query(vector=q_emb, top_k=top_k, include_metadata=True, namespace=namespace)
    docs = []
    for m in res.matches:
        md = m.metadata or {}
        docs.append({
            "score": float(m.score),
            "source": md.get("source"),
            "page": md.get("page"),
            "type": md.get("type"),
            "text": md.get("text", "")
        })
    return docs

def build_prompt(user_q: str, passages: List[dict]) -> list:
    context = "\n\n".join([f"- ({i+1}) {p['text'][:1100]}" for i, p in enumerate(passages)])
    sys = (
        "너는 SSAFY 교육생과 페어프로그래밍하는 AI 동료다. 한국어로 간결히 답하고, "
        "정답만 주지 말고 '힌트 → 풀이' 순서로 설명해줘. "
        "출처에 대한 명시는 절대 하지마. 참고 파일로 넣어둔 커리큘럼이나 명세서에 대해서 그대로 보여주면 안돼. "
        "모르면 모른다고 말해라."
    )
    usr = f"[질문]\n{user_q}\n\n[참고자료]\n{context}"
    return [{"role": "system", "content": sys}, {"role": "user", "content": usr}]

@app.post("/query", response_model=QueryRes, dependencies=[Depends(verify_api_key)])
def rag_query(req: QueryReq):
    passages = retrieve(req.query, req.namespace or "specs", req.top_k)
    msgs = build_prompt(req.query, passages)
    chat = oai.chat.completions.create(model=GEN_MODEL, messages=msgs, temperature=0.2)
    answer = chat.choices[0].message.content
    out = {"answer": answer}
    if req.include_sources:
        out["sources"] = [
            {"i": i+1, "score": p["score"], "source": p["source"], "page": p["page"], "type": p.get("type")}
            for i, p in enumerate(passages)
        ]
    return out
