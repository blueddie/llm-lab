"""
기본 RAG 파이프라인 v1 - 처음부터 직접 구현 (LangChain 없이)

의도적으로 가장 단순한 방법으로 만들었다.
각 단계의 한계를 직접 체감한 뒤, 하나씩 개선해나갈 것이다.

파이프라인:
  [Indexing]  문서 로드 → 고정 크기 청킹 → 임베딩 → 저장
  [Querying]  질문 → 임베딩 → 코사인 유사도 검색 → LLM 생성
  [Eval]      생성된 답변 vs 정답 비교
"""
import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================================
# 설정
# ============================================================
DATA_PATH = Path(__file__).parent / "data" / "korquad_v2_subset.json"
INDEX_PATH = Path(__file__).parent / "data" / "index.npz"  # 임베딩 캐시

CHUNK_SIZE = 500       # 청크 크기 (글자 수)
CHUNK_OVERLAP = 100    # 청크 간 겹치는 부분
TOP_K = 3              # 검색할 청크 수
NUM_DOCS = 20          # 사용할 문서 수 (비용 절약, 빠른 반복)
NUM_EVAL = 50          # 평가할 질문 수
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"


# ============================================================
# 1단계: 데이터 로드
# ============================================================
def load_documents(path: Path, num_docs: int) -> list[dict]:
    """
    저장해둔 KorQuAD 2.0 데이터를 로드한다.

    반환 형태:
    [
        {
            "title": "문서 제목",
            "context_text": "전처리된 텍스트...",
            "qas": [{"question": "...", "answer_text": "..."}]
        },
        ...
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = data["documents"][:num_docs]
    print(f"문서 {len(docs)}개 로드 (전체 {data['num_docs']}개 중)")
    return docs


# ============================================================
# 2단계: 청킹 (Fixed-Size Chunking)
# ============================================================
# 왜 청킹을 하는가?
# - 문서 평균 17,000자. 임베딩 모델에 통째로 넣으면 의미가 뭉개진다.
# - "이 문서는 대략 이런 내용"이라는 뭉뚱그린 벡터가 되어버림.
# - 작은 단위로 쪼개야 "이 부분은 구체적으로 이 내용"이라는 정밀한 벡터가 만들어진다.
#
# 왜 overlap을 주는가?
# - 500자에서 뚝 자르면 문장이 중간에 잘릴 수 있다.
# - 겹침을 주면 잘린 문장이 다음 청크에서 완전한 형태로 포함된다.
# - 단, overlap이 너무 크면 중복이 많아져서 검색 효율이 떨어진다.

def chunk_document(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    텍스트를 고정 크기로 자른다. 가장 단순한 청킹 방법.

    한계 (나중에 개선할 것):
    - 문장 중간에서 잘릴 수 있다
    - 의미 단위를 고려하지 않는다
    - 표나 리스트 구조가 깨질 수 있다
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # 빈 청크는 건너뛴다
        if chunk.strip():
            chunks.append(chunk)

        # 다음 시작점 = 현재 시작 + (청크 크기 - 겹침)
        start += chunk_size - overlap

    return chunks


def chunk_all_documents(docs: list[dict]) -> tuple[list[str], list[dict]]:
    """
    모든 문서를 청킹하고, 각 청크의 출처 정보를 함께 저장한다.

    반환:
    - chunks: 청크 텍스트 리스트
    - metadata: 각 청크의 출처 정보 (어떤 문서의 몇 번째 청크인지)
    """
    chunks = []
    metadata = []

    for doc_idx, doc in enumerate(docs):
        doc_chunks = chunk_document(doc["context_text"], CHUNK_SIZE, CHUNK_OVERLAP)

        for chunk_idx, chunk in enumerate(doc_chunks):
            chunks.append(chunk)
            metadata.append({
                "doc_idx": doc_idx,
                "chunk_idx": chunk_idx,
                "title": doc["title"],
            })

    print(f"총 {len(chunks)}개 청크 생성 (문서당 평균 {len(chunks)/len(docs):.0f}개)")
    return chunks, metadata


# ============================================================
# 3단계: 임베딩
# ============================================================
# 임베딩 프로젝트에서 이미 배운 내용이다.
# 새로운 점: 수천 개의 텍스트를 배치로 임베딩해야 한다.
# OpenAI API는 한 번에 최대 2048개까지 배치 처리 가능.

async def embed_texts(texts: list[str], model: str = EMBEDDING_MODEL) -> np.ndarray:
    """
    텍스트 리스트를 임베딩 벡터로 변환한다.
    OpenAI API의 배치 처리를 활용해서 효율적으로 호출.
    """
    embeddings = []
    batch_size = 1000  # API 배치 제한 고려

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = await client.embeddings.create(
            model=model,
            input=batch,
        )
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
        print(f"  임베딩 완료: {min(i + batch_size, len(texts))}/{len(texts)}")

    return np.array(embeddings, dtype=np.float32)


async def build_index(chunks: list[str], metadata: list[dict]) -> np.ndarray:
    """
    청크들을 임베딩해서 인덱스를 만든다.
    이미 만들어둔 인덱스가 있으면 디스크에서 로드한다 (비용 절약).
    """
    if INDEX_PATH.exists():
        print("기존 인덱스 로드 중...")
        data = np.load(INDEX_PATH, allow_pickle=True)
        return data["embeddings"]

    print(f"{len(chunks)}개 청크 임베딩 중...")
    start = time.time()
    embeddings = await embed_texts(chunks)
    elapsed = time.time() - start

    # 디스크에 저장 (다음 실행 시 재사용)
    np.savez(INDEX_PATH, embeddings=embeddings, metadata=np.array(metadata))
    print(f"인덱스 저장 완료 ({elapsed:.1f}초 소요)")

    return embeddings


# ============================================================
# 4단계: 검색 (Retrieval)
# ============================================================
# 핵심: 질문 벡터와 모든 청크 벡터 사이의 코사인 유사도를 계산하고,
# 가장 유사한 top-k개를 반환한다.

def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """코사인 유사도 계산 (벡터화된 연산으로 한번에)"""
    # 정규화
    query_norm = query_vec / np.linalg.norm(query_vec)
    doc_norms = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    # 내적 = 코사인 유사도 (정규화된 벡터끼리)
    return doc_norms @ query_norm


async def retrieve(
    question: str,
    chunk_embeddings: np.ndarray,
    chunks: list[str],
    metadata: list[dict],
    top_k: int = TOP_K,
) -> list[dict]:
    """
    질문과 가장 관련 있는 청크 top_k개를 찾는다.

    반환:
    [
        {"text": "청크 내용", "score": 0.85, "title": "문서 제목", ...},
        ...
    ]
    """
    # 질문을 임베딩
    query_embedding = await embed_texts([question])
    query_vec = query_embedding[0]

    # 코사인 유사도 계산
    scores = cosine_similarity(query_vec, chunk_embeddings)

    # 상위 k개 인덱스
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "text": chunks[idx],
            "score": float(scores[idx]),
            "title": metadata[idx]["title"],
            "doc_idx": metadata[idx]["doc_idx"],
            "chunk_idx": metadata[idx]["chunk_idx"],
        })

    return results


# ============================================================
# 5단계: 생성 (Generation)
# ============================================================
# 검색된 청크들을 프롬프트에 넣어서 LLM에게 답변을 요청한다.
# 프롬프트 설계가 답변 품질에 큰 영향을 미친다.

SYSTEM_PROMPT = """당신은 주어진 참고 자료를 기반으로 질문에 답변하는 어시스턴트입니다.

규칙:
1. 반드시 참고 자료에 있는 정보만 사용해서 답변하세요.
2. 참고 자료에 답이 없으면 "참고 자료에서 답을 찾을 수 없습니다."라고 답하세요.
3. 답변은 간결하게, 핵심만 말하세요.
"""


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """검색된 청크들로 프롬프트를 조립한다."""
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(f"[참고자료 {i}] (출처: {chunk['title']}, 유사도: {chunk['score']:.3f})\n{chunk['text']}")

    context = "\n\n".join(context_parts)

    return f"""참고 자료:
{context}

질문: {question}

답변:"""


async def generate(question: str, retrieved_chunks: list[dict]) -> str:
    """검색된 청크를 컨텍스트로 넣어서 LLM이 답변을 생성한다."""
    prompt = build_prompt(question, retrieved_chunks)

    response = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,  # 일관된 답변을 위해
        max_tokens=200,
    )

    return response.choices[0].message.content


# ============================================================
# 6단계: 평가 (Evaluation)
# ============================================================
# 가장 단순한 평가: 정답 텍스트가 생성된 답변에 포함되어 있는가?
# 이것을 "Exact Inclusion"이라 부르자.
#
# 한계: "332cm"와 "332 cm"은 같은 답인데 다르다고 판단할 수 있다.
# 나중에 더 정교한 평가 방법(RAGAS 등)으로 개선할 것이다.

def evaluate_answer(generated: str, ground_truth: str) -> dict:
    """
    생성된 답변과 정답을 비교한다.

    평가 지표:
    - exact_inclusion: 정답이 생성 답변에 포함되는가
    - retrieval_hit: 검색된 청크에 정답이 포함되어 있는가 (검색 자체의 성능)
    """
    # 공백/줄바꿈 정규화
    generated_clean = " ".join(generated.split())
    truth_clean = " ".join(ground_truth.split())

    return {
        "exact_inclusion": truth_clean in generated_clean,
    }


def evaluate_retrieval(retrieved_chunks: list[dict], ground_truth: str) -> bool:
    """검색된 청크들 중에 정답이 포함되어 있는가? (검색 성능 평가)"""
    truth_clean = " ".join(ground_truth.split())
    for chunk in retrieved_chunks:
        chunk_clean = " ".join(chunk["text"].split())
        if truth_clean in chunk_clean:
            return True
    return False


# ============================================================
# 메인 파이프라인
# ============================================================
async def main():
    print("=" * 60)
    print("기본 RAG 파이프라인 v1")
    print("=" * 60)

    # 1. 데이터 로드
    docs = load_documents(DATA_PATH, NUM_DOCS)

    # 2. 청킹
    chunks, metadata = chunk_all_documents(docs)

    # 3. 인덱스 생성 (임베딩)
    embeddings = await build_index(chunks, metadata)
    print(f"인덱스 크기: {embeddings.shape} (청크 수 × 벡터 차원)")

    # 4. 평가용 QA 수집
    eval_qas = []
    for doc_idx, doc in enumerate(docs):
        for qa in doc["qas"]:
            eval_qas.append({
                "question": qa["question"],
                "answer": qa["answer_text"],
                "title": doc["title"],
                "doc_idx": doc_idx,
            })
    eval_qas = eval_qas[:NUM_EVAL]
    print(f"\n평가 대상: {len(eval_qas)}개 질문")

    # 5. RAG 실행 & 평가
    print("\n" + "-" * 60)
    print("RAG 실행 중...")
    print("-" * 60)

    retrieval_hits = 0
    generation_hits = 0

    for i, qa in enumerate(eval_qas):
        # 검색
        retrieved = await retrieve(
            qa["question"], embeddings, chunks, metadata, TOP_K
        )

        # 검색 평가: 정답이 검색된 청크에 있는가?
        retrieval_hit = evaluate_retrieval(retrieved, qa["answer"])
        retrieval_hits += retrieval_hit

        # 생성
        answer = await generate(qa["question"], retrieved)

        # 생성 평가: 정답이 생성된 답변에 포함되는가?
        eval_result = evaluate_answer(answer, qa["answer"])
        generation_hits += eval_result["exact_inclusion"]

        # 진행 상황 출력 (10개마다 상세, 나머지는 간략)
        if i < 3 or i % 10 == 0:
            print(f"\n[{i+1}/{len(eval_qas)}] {qa['title']}")
            print(f"  Q: {qa['question']}")
            print(f"  정답: {qa['answer'][:80]}")
            print(f"  생성: {answer[:80]}")
            print(f"  검색 히트: {'O' if retrieval_hit else 'X'} | 생성 정확: {'O' if eval_result['exact_inclusion'] else 'X'}")
        else:
            status = "O" if eval_result["exact_inclusion"] else "X"
            print(f"  [{i+1}] {status} - {qa['question'][:40]}...")

    # 6. 최종 결과
    print("\n" + "=" * 60)
    print("최종 결과")
    print("=" * 60)
    print(f"평가 질문 수: {len(eval_qas)}")
    print(f"검색 정확도 (Retrieval Hit Rate): {retrieval_hits}/{len(eval_qas)} ({retrieval_hits/len(eval_qas)*100:.1f}%)")
    print(f"생성 정확도 (Exact Inclusion):    {generation_hits}/{len(eval_qas)} ({generation_hits/len(eval_qas)*100:.1f}%)")
    print()
    print("해석:")
    print(f"  - 검색에서 정답 청크를 찾은 비율: {retrieval_hits/len(eval_qas)*100:.1f}%")
    print(f"  - 최종 답변에 정답이 포함된 비율: {generation_hits/len(eval_qas)*100:.1f}%")
    if retrieval_hits > generation_hits:
        print(f"  - 검색은 했는데 생성에서 놓친 경우: {retrieval_hits - generation_hits}건 → 프롬프트 개선 필요")
    if retrieval_hits < len(eval_qas) * 0.5:
        print(f"  - 검색 정확도가 낮음 → 청킹 전략 또는 검색 방법 개선 필요")


if __name__ == "__main__":
    asyncio.run(main())

'''
============================================================
기본 RAG 파이프라인 v1
============================================================
문서 20개 로드 (전체 200개 중)
총 1493개 청크 생성 (문서당 평균 75개)
1493개 청크 임베딩 중...
  임베딩 완료: 1000/1493
  임베딩 완료: 1493/1493
인덱스 저장 완료 (7.7초 소요)
인덱스 크기: (1493, 1536) (청크 수 × 벡터 차원)

평가 대상: 50개 질문

------------------------------------------------------------
RAG 실행 중...
------------------------------------------------------------
  임베딩 완료: 1/1

[1/50] 이명박
  Q: 이명박이 서울광장을 추진한 과정은 어땠어?
  정답: 이명박은 서울시장 취임 후 서울시청 앞에 광장을 조성하고자 했다. 그러나 서울시청 앞은 교통이 가장 복잡한 지역이었다. 시청 앞 광장 조성은 꼭
  생성: 이명박은 서울시청 앞에 광장을 조성하고자 했으나, 교통체증 우려로 반대가 많았다. 그는 시뮬레이션을 통해 교통상황을 검토한 결과, 크게 악화되지
  검색 히트: X | 생성 정확: X
  임베딩 완료: 1/1

[2/50] 이명박
  Q: 2011년 1월 발생한 삼호쥬얼리호 구출 작전의 작전명은 무엇이었습니까?
  정답: 아덴만 여명작전
  생성: 작전명은 '아덴만 여명작전'입니다.
  검색 히트: O | 생성 정확: O
  임베딩 완료: 1/1

[3/50] 이명박
  Q: 미국발 글로벌금융위기는 언제 일어났나?
  정답: 2008년 9월
  생성: 참고 자료에서 답을 찾을 수 없습니다.
  검색 히트: X | 생성 정확: X
  임베딩 완료: 1/1
  [4] O - 어느 대통령 집권때 광우병 사태가 발생하였나?...
  임베딩 완료: 1/1
  [5] O - 한미 쇠고기협상 타결 이후 시위가 일어나자 이명박이 수입 금지 재협상을 ...
  임베딩 완료: 1/1
  [6] X - 2008년 글로벌금융위기가 발생하자 외신이 아시아에서 가장 먼저 국가부도...
  임베딩 완료: 1/1
  [7] X - 이명박이 경제위기 극복과정에서 어려움에 처한 중소기업과 서민 살리기 행보...
  임베딩 완료: 1/1
  [8] X - 미국산 쇠고기의 위험성을 보도한 MBC 프로그램 제목은?...
  임베딩 완료: 1/1
  [9] O - 광우병 사태로 인해 이명박 대통령은 몇 개월 동안 소고기 수입을 금지하였...
  임베딩 완료: 1/1
  [10] X - 이명박 정권 기간 중 미국발 글로벌금융위기가 발생한 시기는?...
  임베딩 완료: 1/1

[11/50] 이명박
  Q: 한국은 언제 신용평가사로부터 가장 높은 국가신용등급을 받았는가?
  정답: 2012년 8월
  생성: 한국은 2012년 8월에 신용평가사로부터 가장 높은 국가신용등급을 받았습니다.
  검색 히트: O | 생성 정확: O
  임베딩 완료: 1/1
  [12] O - 2015년 4대강 사업에 대한 적법 판결을 어디서 시행하였나?...
  임베딩 완료: 1/1
  [13] O - 2012년 한국에 중국과 일본을 앞서는 국가신용등급을 준 신용평가사는?...
  임베딩 완료: 1/1
  [14] O - 논란이 된 4대강 사업이 대법원으로부터 적법 판결을 받은 해는?...
  임베딩 완료: 1/1
  [15] X - 금강산 관광객 피격 사망 사건, 북핵문제 등을 겪으면서 이명박 정부거 선...
  임베딩 완료: 1/1
  [16] O - 미국 국가신용등급 강등으로 전 세계가 충격에 빠진 시기는?...
  임베딩 완료: 1/1
  [17] X - 한국이 사상 처음으로 일본과 중국을 앞지른 국가신용등급은 누구로부터 받은...
  임베딩 완료: 1/1
  [18] O - 이명박 정권의 4대강 사업이 대법원에 의해 적법 판결을 받은 시기는?...
  임베딩 완료: 1/1
  [19] O - 한국은 UAE 원전을 언제 처음 수주하였나?...
  임베딩 완료: 1/1
  [20] X - 우리나라는 세계에서 몇번째로 원전수출을 하였나?...
  임베딩 완료: 1/1

[21/50] 이명박
  Q: 외교정책에 있어 이명박 정부가 국정지표로 삼은 것은?
  정답: 더 큰 대한민국
  생성: 이명박 정부는 '작은 정부, 큰 시장'을 큰 뼈대로 하여 '경제살리기'를 국정지표로 삼았습니다.
  검색 히트: X | 생성 정확: X
  임베딩 완료: 1/1
  [22] O - UAE 원전 수주로 한국이 세계 6번째 원전 수출국에 올라선 해는?...
  임베딩 완료: 1/1
  [23] X - 이명박 정부가 일본으로부터 돌려받은 조선 시대 서적은?...
  임베딩 완료: 1/1
  [24] X - 이명박의 외교정책에서 국정지표로 삼은 모토는 무엇인가?...
  임베딩 완료: 1/1
  [25] O - 이명박 정권 시절 서울 G20 정상회의가 열린 시기는 언제인가?...
  임베딩 완료: 1/1
  [26] O - 이명박 정권 시절 UAE의 원자력발전소를 수주하면서 세계 몇 번째의 원전...
  임베딩 완료: 1/1
  [27] O - 이명박의 아버지 이름은?...
  임베딩 완료: 1/1
  [28] O - 이명박의 개명하기 전 이름은?...
  임베딩 완료: 1/1
  [29] O - 이명박의 개명 전 이름은?...
  임베딩 완료: 1/1
  [30] O - 이명박이 창씨 개명한 이름은?...
  임베딩 완료: 1/1

[31/50] 이명박
  Q: 이명박 대통령의 아버지의 이름은 무엇인가?
  정답: 이충우
  생성: 이명박 대통령의 아버지의 이름은 이충우입니다.
  검색 히트: O | 생성 정확: O
  임베딩 완료: 1/1
  [32] O - 이충우의 고향은 어디인가?...
  임베딩 완료: 1/1
  [33] O - 이충우의 본명은 무엇인가?...
  임베딩 완료: 1/1
  [34] X - 서울로 온 이명박이 대학중퇴라도 되고 싶다는 마음에 대입관련 책을 얻어온...
  임베딩 완료: 1/1
  [35] X - 이명박은 고등학교 졸업 후 어디로 이사하였는가?...
  임베딩 완료: 1/1
  [36] X - 1961년 이명박이 입학한 대학교 이름은?...
  임베딩 완료: 1/1
  [37] X - 이명박은 15대 총선에서 어느 지역구에 출마했는가?...
  임베딩 완료: 1/1
  [38] O - 이명박 캠프 선거 비용이 법정비용을 초과했다고 폭로한 사람의 이름은?...
  임베딩 완료: 1/1
  [39] O - 1996년 15대 총선에서 이명박이 출마한 지역구는?...
  임베딩 완료: 1/1
  [40] O - 이명박 캠프의 선거비용이 법정비용을 초과했다고 폭로한 인물은?...
  임베딩 완료: 1/1

[41/50] 이명박
  Q: 이명박 측이 김유찬에게 3억 원을 주고 폭로를 지시했다고 주장한 인물은?
  정답: 이종찬
  생성: 이명박 측이 김유찬에게 3억 원을 주고 폭로를 지시했다고 주장한 인물은 이종찬 후보입니다.
  검색 히트: O | 생성 정확: O
  임베딩 완료: 1/1
  [42] O - 1996년 15대 총선에서 이명박이 출마한 지역구는?...
  임베딩 완료: 1/1
  [43] O - 이명박 캠프의 선거비용과 관련하여 기자회견을 열어 폭로한 사람은 누구인가...
  임베딩 완료: 1/1
  [44] O - 김유찬은 이종찬으로부터 얼마를 받았다고 주장하였는가?...
  임베딩 완료: 1/1
  [45] X - 타임지는 어느 이유로 이명박을 환경영웅 중 한명으로 선정하였나?...
  임베딩 완료: 1/1
  [46] O - 2004년 이명박이 서울시 대중교통의 체계를 전면 개편하면서 신설된 교통...
  임베딩 완료: 1/1
  [47] X - 이명박의 교통체재 개편을 교통혁명에 비견하며 우수정책으로 인정한 곳은?...
  임베딩 완료: 1/1
  [48] O - 이명박의 서울시 교통체계 개편을 교통혁명에 비유하며 인정한 조직은 무엇인...
  임베딩 완료: 1/1
  [49] O - 이명박은 누구와 BBK를 설립하였나?...
  임베딩 완료: 1/1
  [50] O - 검찰은 BBK 주가 조작 사건에 대해 이명박은 어떤 혐의를 받았는가?...

============================================================
최종 결과
============================================================
평가 질문 수: 50
검색 정확도 (Retrieval Hit Rate): 35/50 (70.0%)
생성 정확도 (Exact Inclusion):    32/50 (64.0%)

해석:
  - 검색에서 정답 청크를 찾은 비율: 70.0%
  - 최종 답변에 정답이 포함된 비율: 64.0%
  - 검색은 했는데 생성에서 놓친 경우: 3건 → 프롬프트 개선 필요
'''