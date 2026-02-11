"""
POST 요청 + Pydantic Model

GET: URL에 파라미터 (/add?a=1&b=2)
POST: body에 JSON 데이터를 담아서 전송

Pydantic Model = 요청/응답 데이터의 "설계도"
- 어떤 필드가 있어야 하는지
- 각 필드의 타입은 무엇인지
- 필수인지 선택인지
→ 조건에 안 맞으면 FastAPI가 자동으로 에러 반환
"""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


# --- 요청/응답 모델 정의 ---

class ChatRequest(BaseModel):
    """채팅 요청 — OpenAI API 요청 body의 간소화 버전"""
    message: str                    # 필수
    model: str = "gpt-4.1-nano"    # 선택 (기본값 있음)
    max_tokens: int = 100           # 선택 (기본값 있음)


class ChatResponse(BaseModel):
    """채팅 응답"""
    reply: str
    model_used: str
    tokens_used: int


# --- 엔드포인트 ---

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    POST /chat — 채팅 요청을 받아서 응답 반환

    request 파라미터의 타입이 ChatRequest
    → FastAPI가 자동으로:
      1. JSON body를 파싱
      2. ChatRequest 모델에 맞는지 검증
      3. 안 맞으면 422 에러 + 무엇이 틀렸는지 알려줌

    response_model=ChatResponse
    → 응답도 이 형태로 자동 검증 + docs에 표시
    """

    # 지금은 실제 LLM 호출 대신 가짜 응답 (나중에 연결할 예정)
    fake_reply = f"'{request.message}'에 대한 응답입니다. (모델: {request.model})"

    return ChatResponse(
        reply=fake_reply,
        model_used=request.model,
        tokens_used=len(request.message) + len(fake_reply),  # 가짜 토큰 수
    )


# --- 실무 패턴: 여러 메시지를 한번에 ---

class BatchRequest(BaseModel):
    """배치 요청 — 여러 메시지를 한번에 처리"""
    messages: list[str]             # 문자열 리스트
    model: str = "gpt-4.1-nano"


class BatchResponse(BaseModel):
    """배치 응답"""
    results: list[ChatResponse]
    total_count: int


@app.post("/chat/batch", response_model=BatchResponse)
async def chat_batch(request: BatchRequest):
    """
    POST /chat/batch — 여러 메시지를 한번에 처리

    list[str], list[ChatResponse] 같은 타입 힌트가
    Pydantic에서는 "이 필드는 문자열 리스트여야 해"라는 검증 규칙이 됨
    """

    results = []
    for msg in request.messages:
        fake_reply = f"'{msg}'에 대한 응답입니다."
        results.append(ChatResponse(
            reply=fake_reply,
            model_used=request.model,
            tokens_used=len(msg) + len(fake_reply),
        ))

    return BatchResponse(
        results=results,
        total_count=len(results),
    )
