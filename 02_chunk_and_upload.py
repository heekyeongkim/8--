"""
계층적 청킹 → OpenAI 임베딩 → Supabase 적재

구조:
  대청크 (부모): ❶~❻ 항목 전체
  소청크 (자식): 각 항목 안의 ○ 세부 항목

pip install openai supabase python-docx python-dotenv
"""

import os
import re
from docx import Document
from openai import OpenAI
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SUPABASE_URL    = os.getenv("SUPABASE_URL")
SUPABASE_KEY    = os.getenv("SUPABASE_KEY")
DOCX_PATH       = "정부청사관리본부_개인정보_내부관리계획_개정_계획.docx"
EMBED_MODEL     = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
BATCH_SIZE      = 20
TABLE_NAME      = "privacy_plan_chunks"

openai_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)
supabase      = create_client(SUPABASE_URL, SUPABASE_KEY)


# ─────────────────────────────────────────
# 1. 문서 추출
# ─────────────────────────────────────────
def extract_paragraphs(path: str) -> list[str]:
    doc = Document(path)
    return [p.text.strip() for p in doc.paragraphs if p.text.strip()]


# ─────────────────────────────────────────
# 2. 계층적 청킹
#    부모: ❶❷❸❹❺❻ 항목
#    자식: 각 항목 안의 ○ 세부 내용
# ─────────────────────────────────────────
def hierarchical_chunking(paragraphs: list[str]) -> list[dict]:
    chunks      = []
    parent_text = None
    parent_title = None
    child_buf   = []
    section_no  = 0

    # ❶~❻ 탐지 패턴
    PARENT_PATTERN = re.compile(r'^[❶❷❸❹❺❻]')
    # ○ 세부항목 탐지
    CHILD_PATTERN  = re.compile(r'^[○]')

    def flush_child(buf, parent_t, parent_title, section_no, child_no):
        if not buf:
            return None
        text = " ".join(buf)
        return {
            "content":    text,
            "parent":     parent_t,
            "chunk_type": "child",
            "section_no": section_no,
            "child_no":   child_no,
            "section_title": parent_title,
        }

    child_no = 0

    for para in paragraphs:
        if PARENT_PATTERN.match(para):
            # 이전 자식 청크 저장
            if child_buf and parent_text:
                c = flush_child(child_buf, parent_text, parent_title, section_no, child_no)
                if c:
                    chunks.append(c)
                child_buf = []
                child_no  = 0

            # 새 부모 청크 시작
            section_no  += 1
            parent_title = para
            parent_text  = para

            # 부모 청크 자체도 저장
            chunks.append({
                "content":       para,
                "parent":        None,
                "chunk_type":    "parent",
                "section_no":    section_no,
                "child_no":      0,
                "section_title": para,
            })

        elif CHILD_PATTERN.match(para) and parent_text:
            # 이전 자식 저장
            if child_buf:
                c = flush_child(child_buf, parent_text, parent_title, section_no, child_no)
                if c:
                    chunks.append(c)
            child_no += 1
            child_buf = [para]

        elif parent_text and child_buf:
            # 세부 내용 (-, ※ 등) 현재 자식에 붙임
            child_buf.append(para)

        else:
            # 배경, 추진일정 등 기타 텍스트
            if len(para) > 15:
                chunks.append({
                    "content":       para,
                    "parent":        None,
                    "chunk_type":    "other",
                    "section_no":    0,
                    "child_no":      0,
                    "section_title": "기타",
                })

    # 마지막 자식 저장
    if child_buf and parent_text:
        c = flush_child(child_buf, parent_text, parent_title, section_no, child_no)
        if c:
            chunks.append(c)

    print(f"[2] 계층적 청킹 완료: 총 {len(chunks)}개")
    for ch in chunks:
        tag = f"[{ch['chunk_type']}] 섹션{ch['section_no']}-{ch['child_no']}"
        print(f"    {tag}: {ch['content'][:60]}...")
    return chunks


# ─────────────────────────────────────────
# 3. 임베딩
# ─────────────────────────────────────────
def embed_texts(texts: list[str]) -> list[list[float]]:
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch    = texts[i : i + BATCH_SIZE]
        response = openai_client.embeddings.create(model=EMBED_MODEL, input=batch, encoding_format="float")
        all_embeddings.extend([r.embedding for r in response.data])
        print(f"  임베딩: {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")
    return all_embeddings


# ─────────────────────────────────────────
# 4. Supabase 적재
# ─────────────────────────────────────────
def upload(chunks: list[dict], embeddings: list[list[float]]):
    records = []
    for chunk, emb in zip(chunks, embeddings):
        records.append({
            "content":   chunk["content"],
            "parent":    chunk["parent"],
            "metadata":  {
                "chunk_type":    chunk["chunk_type"],
                "section_no":    chunk["section_no"],
                "child_no":      chunk["child_no"],
                "section_title": chunk["section_title"],
                "source":        "정부청사관리본부_개인정보_내부관리계획_개정_계획",
            },
            "embedding": emb,
        })

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        supabase.table(TABLE_NAME).upsert(batch).execute()
        print(f"  적재: {min(i + BATCH_SIZE, len(records))}/{len(records)}")

    print(f"[4] ✅ 적재 완료 → {TABLE_NAME}")


# ─────────────────────────────────────────
# 실행
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("[1] 문서 추출 중...")
    paragraphs = extract_paragraphs(DOCX_PATH)

    chunks = hierarchical_chunking(paragraphs)

    print("\n[3] 임베딩 생성 중...")
    texts      = [c["content"] for c in chunks]
    embeddings = embed_texts(texts)

    print("\n[4] Supabase 적재 중...")
    upload(chunks, embeddings)
