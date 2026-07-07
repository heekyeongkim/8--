# 8회차 - RAG 파이프라인 실습

<img width="578" height="803" alt="image" src="https://github.com/user-attachments/assets/464feefb-e500-4f00-bd9a-f637767092f0" />

문서(docx)를 계층적으로 청킹하여 Supabase(pgvector)에 임베딩 적재하고,
Naive RAG / Advanced RAG(Rerank 포함) 방식으로 질의응답하는 실습 프로젝트입니다.

## 구성 파일

| 파일 | 설명 |
|---|---|
| [01_setup_supabase.sql](01_setup_supabase.sql) | Supabase 테이블(`privacy_plan_chunks`) 및 벡터 검색 함수 생성 |
| [02_chunk_and_upload.py](02_chunk_and_upload.py) | docx 문서를 계층적 청킹(부모/자식) 후 임베딩하여 Supabase에 적재 |
| [03_rag_pipeline.py](03_rag_pipeline.py) | Naive RAG / Advanced RAG(Cohere Rerank) 질의응답 파이프라인 |
| [env.example](env.example) | 필요한 환경변수 템플릿 |

## 사용 기술

- **임베딩**: OpenRouter (`nvidia/llama-nemotron-embed-vl-1b-v2:free`)
- **벡터 DB**: Supabase (pgvector)
- **Re-rank**: Cohere `rerank-v3.5`
- **생성 모델**: OpenRouter (`nvidia/nemotron-3-super-120b-a12b:free`)

## 실행 방법

```bash
pip install openai supabase python-docx python-dotenv cohere requests

cp env.example .env
# .env에 실제 API 키 입력 (커밋 금지)

python 02_chunk_and_upload.py   # 문서 청킹 → 임베딩 → Supabase 적재
python 03_rag_pipeline.py       # Naive/Advanced RAG 질의응답 비교
```

> ⚠️ `.env`나 API 키가 담긴 파일은 절대 커밋하지 마세요.

## 참고

- Artifact: https://claude.ai/code/artifact/bcddcfd4-8ef1-4f5f-bd56-e736a0da9714?via=auto_preview
