-- =============================================
-- 개인정보 내부관리계획 RAG 파이프라인
-- 기존 테이블(documents, documents_hierarchical, privacy_policy_docs)과 겹치지 않는 이름
-- =============================================

-- 0. pgvector 확장 활성화 (이미 켜져 있으면 무시됨)
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. 테이블 생성
CREATE TABLE IF NOT EXISTS privacy_plan_chunks (
    id            bigserial PRIMARY KEY,
    content       text        NOT NULL,       -- 청크 본문
    parent        text,                        -- 부모 청크 (상위 항목 전체 텍스트)
    metadata      jsonb,                       -- 구조 정보
    embedding     vector(2048)                 -- nvidia/llama-nemotron-embed-vl-1b-v2
);

-- 2. 벡터 인덱스
-- 주의: ivfflat/hnsw 인덱스는 pgvector에서 최대 2000차원까지만 지원됨.
-- 이 프로젝트의 임베딩 모델(nvidia/llama-nemotron-embed-vl-1b-v2)은 2048차원이라 인덱스 생성 불가.
-- 데이터 규모가 작으므로(수십 건) 인덱스 없이 순차 스캔으로도 충분함.

-- 3. Naive RAG 검색 함수 (단순 벡터 유사도)
CREATE OR REPLACE FUNCTION search_privacy_plan_naive(
    query_embedding vector(2048),
    match_threshold float DEFAULT 0.3,
    match_count     int   DEFAULT 10
)
RETURNS TABLE (
    id          bigint,
    content     text,
    parent      text,
    metadata    jsonb,
    similarity  float
)
LANGUAGE sql STABLE AS $$
    SELECT
        privacy_plan_chunks.id,
        privacy_plan_chunks.content,
        privacy_plan_chunks.parent,
        privacy_plan_chunks.metadata,
        1 - (privacy_plan_chunks.embedding <=> query_embedding) AS similarity
    FROM privacy_plan_chunks
    WHERE 1 - (privacy_plan_chunks.embedding <=> query_embedding) > match_threshold
    ORDER BY privacy_plan_chunks.embedding <=> query_embedding
    LIMIT match_count;
$$;

-- 4. Advanced RAG 검색 함수 (부모 청크 포함 반환)
CREATE OR REPLACE FUNCTION search_privacy_plan_advanced(
    query_embedding vector(2048),
    match_threshold float DEFAULT 0.3,
    match_count     int   DEFAULT 5
)
RETURNS TABLE (
    id          bigint,
    content     text,
    parent      text,
    metadata    jsonb,
    similarity  float
)
LANGUAGE sql STABLE AS $$
    SELECT
        privacy_plan_chunks.id,
        privacy_plan_chunks.content,
        privacy_plan_chunks.parent,   -- 부모 청크를 함께 반환 → 문맥 확장
        privacy_plan_chunks.metadata,
        1 - (privacy_plan_chunks.embedding <=> query_embedding) AS similarity
    FROM privacy_plan_chunks
    WHERE 1 - (privacy_plan_chunks.embedding <=> query_embedding) > match_threshold
        AND privacy_plan_chunks.metadata->>'chunk_type' = 'child'  -- 소청크만 검색
    ORDER BY privacy_plan_chunks.embedding <=> query_embedding
    LIMIT match_count;
$$;
