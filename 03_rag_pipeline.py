"""
RAG 파이프라인 (Naive RAG + Advanced RAG)

- 임베딩: OpenRouter (nvidia/llama-nemotron-embed-vl-1b-v2:free) - 무료
- 검색:   Supabase 벡터 검색
- Re-rank: Cohere rerank
- 생성:   OpenRouter chat 모델

pip install openai supabase cohere python-dotenv requests
"""

import os
import requests
from openai import OpenAI
from supabase import create_client
import cohere
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# 환경변수
# ─────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SUPABASE_URL       = os.getenv("SUPABASE_URL")
SUPABASE_KEY       = os.getenv("SUPABASE_KEY")
COHERE_API_KEY     = os.getenv("COHERE_API_KEY")

# ─────────────────────────────────────────
# 모델 설정
# ─────────────────────────────────────────
EMBED_MODEL   = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
CHAT_MODEL    = "nvidia/nemotron-3-super-120b-a12b:free"
RERANK_MODEL  = "rerank-v3.5"
TABLE_NAME    = "privacy_plan_chunks"
RPC_FUNCTION  = "search_privacy_plan_naive"

# ─────────────────────────────────────────
# 클라이언트
# ─────────────────────────────────────────
openai_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)
supabase      = create_client(SUPABASE_URL, SUPABASE_KEY)
cohere_client = cohere.Client(api_key=COHERE_API_KEY)


def embed_query(query: str) -> list[float]:
    return openai_client.embeddings.create(
        model=EMBED_MODEL, input=[query], encoding_format="float"
    ).data[0].embedding


def rerank(query: str, chunks: list[dict], top_n: int = 3) -> list[dict]:
    docs     = [c["content"] for c in chunks]
    response = cohere_client.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=docs,
        top_n=min(top_n, len(docs)),
    )
    reranked = []
    for r in response.results:
        chunk = chunks[r.index].copy()
        chunk["rerank_score"] = r.relevance_score
        reranked.append(chunk)
    return reranked


def generate_answer(query: str, context: str) -> str:
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type":  "application/json",
        },
        json={
            "model": CHAT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "당신은 정부청사관리본부 개인정보 내부관리계획 전문가입니다. "
                        "제공된 문서 내용을 바탕으로 정확하고 간결하게 답변하세요. "
                        "문서에 없는 내용은 답변하지 마세요."
                    ),
                },
                {
                    "role": "user",
                    "content": f"[참고 문서]\n{context}\n\n[질문]\n{query}",
                },
            ],
        },
    )
    return response.json()["choices"][0]["message"]["content"]


# ─────────────────────────────────────────
# Naive RAG (벡터 검색 → 답변)
# ─────────────────────────────────────────
def naive_rag(query: str) -> dict:
    print("\n" + "="*50)
    print("[ Naive RAG ]")
    print("="*50)

    q_emb  = embed_query(query)
    result = supabase.rpc(RPC_FUNCTION, {
        "query_embedding": q_emb,
        "match_threshold": 0.3,
        "match_count":     3,
    }).execute()
    chunks = result.data
    print(f"검색된 청크: {len(chunks)}개")
    for c in chunks:
        print(f"  [{c['similarity']:.3f}] {c['content'][:60]}...")

    if not chunks:
        return {"mode": "naive", "chunks": [], "answer": "관련 문서를 찾을 수 없습니다."}

    context = "\n\n".join([c["content"] for c in chunks])
    answer  = generate_answer(query, context)
    print(f"\n답변:\n{answer}")

    return {"mode": "naive", "chunks": chunks, "answer": answer}


# ─────────────────────────────────────────
# Advanced RAG (벡터 검색 → Rerank → 답변)
# ─────────────────────────────────────────
def advanced_rag(query: str) -> dict:
    print("\n" + "="*50)
    print("[ Advanced RAG (with Rerank) ]")
    print("="*50)

    q_emb  = embed_query(query)
    result = supabase.rpc(RPC_FUNCTION, {
        "query_embedding": q_emb,
        "match_threshold": 0.3,
        "match_count":     10,
    }).execute()
    chunks = result.data
    print(f"1차 벡터 검색: {len(chunks)}개")

    if not chunks:
        return {"mode": "advanced", "chunks": [], "answer": "관련 문서를 찾을 수 없습니다."}

    reranked = rerank(query, chunks, top_n=3)
    print(f"Rerank 후: {len(reranked)}개")
    for r in reranked:
        print(f"  [{r['rerank_score']:.3f}] {r['content'][:60]}...")

    context = "\n\n".join([r["content"] for r in reranked])
    answer  = generate_answer(query, context)
    print(f"\n답변:\n{answer}")

    return {"mode": "advanced", "chunks": reranked, "answer": answer}


# ─────────────────────────────────────────
# 실행
# ─────────────────────────────────────────
if __name__ == "__main__":
    query = "인터넷망 차단 조치 기준이 어떻게 바뀌었나요?"

    naive_result    = naive_rag(query)
    advanced_result = advanced_rag(query)

    print("\n" + "="*50)
    print("[ 비교 ]")
    print("="*50)
    print(f"\n[Naive]\n{naive_result['answer']}")
    print(f"\n[Advanced]\n{advanced_result['answer']}")
