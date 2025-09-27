# SSAFY AI Pair Programming (RAG 기반)

SSAFY 교육생을 위한 **AI Pair Programming 어시스턴트**  
주요 명세서, 커리큘럼, 대화 예시 자료 등을 벡터 데이터베이스에 저장하여,  
질문에 대한 힌트와 풀이를 제공하는 **Retrieval-Augmented Generation(RAG)** 서비스입니다.

---

## 📌 주요 기능
- **문서 업로드 및 분할**: PDF, CSV, JSONL 파일을 자동 분할/전처리 후 벡터화  
- **벡터 검색**: Pinecone 기반 의미 검색 (Semantic Search)  
- **응답 생성**: OpenAI Chat Completion을 이용해 한국어 기반 힌트 & 풀이 제공  
- **API 제공**: FastAPI 서버를 통해 `/query` 엔드포인트로 질의 가능  
- **배포**: Render를 이용해 외부에서 접근 가능한 API 서버로 제공  

---

## 🚀 기술 스택
- **Backend**: FastAPI, Uvicorn  
- **Vector DB**: Pinecone  
- **Embedding Model**: OpenAI *text-embedding-3-small*  
- **Generation Model**: OpenAI *gpt-4o-mini*  
- **Infra/Deploy**: Render (Serverless Hosting)  

---

## 📂 프로젝트 구조
```
rag-ssafy/
 ├── app/
 │   ├── main.py        # FastAPI 진입점
 │   ├── utils.py       # 문서 로드 & 벡터 생성 함수
 │   └── ...
 ├── data/              # (예시) 로컬 문서 폴더
 ├── requirements.txt   # Python dependencies
 └── README.md
```

---

## ⚙️ 실행 방법

### 1) 로컬 실행
```bash
# 가상환경 생성 & 활성화
python -m venv .venv
source .venv/Scripts/activate  # (Windows PowerShell)
# 또는
source .venv/bin/activate      # (Linux/Mac)

# 패키지 설치
pip install -r requirements.txt

# 서버 실행
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

실행 후 Swagger UI:  
👉 `http://127.0.0.1:8000/docs`

---

### 2) 배포 (Render 예시)
- Render 대시보드에서 **New Web Service** → GitHub repo 연결  
- Start Command:  
```bash
uvicorn app.main:app --host 0.0.0.0 --port 10000
```
- 배포 완료 후 발급된 URL 예시:  
👉 `https://ai-pair-programming-rag.onrender.com`

---

## 🔗 API 예시

### 요청
```bash
curl -X POST "https://ai-pair-programming-rag.onrender.com/query"   -H "Content-Type: application/json"   -H "X-API-Key: <INTERNAL_API_KEY>"   -d '{"query":"여행 경로 기능 요구사항 요약","namespace":"specs","top_k":3}'
```

### 응답
```json
{
  "answer": "여행 경로 기능은 사용자가 ...",
  "sources": [
    {"i":1,"score":0.89,"source":"관통2.FrontEnd_PJT.pdf","page":3,"type":"pdf"}
  ]
}
```

---

## 📜 개인정보 보호 정책
- 이 프로젝트는 데모용으로, 사용자 데이터를 별도로 저장하지 않습니다.  
- API Key는 반드시 `.env` 파일 또는 Render 환경 변수에 설정해야 하며, GitHub에는 절대 올리지 마세요.  
