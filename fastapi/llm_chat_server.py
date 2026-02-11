"""
실제 LLM API를 호출하는 채팅 서버

조합하는 것들:
- FastAPI — API 서버
- httpx.AsyncClient — 비동기 HTTP 호출
- Pydantic — 요청/응답 검증
- Lifespan — 앱 생명주기에서 Client 관리

이전 실습에서는 "호출하는 쪽"을 만들었고,
이번에는 "받는 쪽 서버"가 다시 "LLM API를 호출하는 쪽"이 되는 구조:

사용자 → [이 서버] → OpenAI API
        FastAPI      httpx
"""

import os
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

OPENAI_URL = "https://api.openai.com/v1/chat/completions"


# --- Lifespan: 앱 시작/종료 시 실행되는 코드 ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    앱의 생명주기를 관리

    yield 기준으로:
    - 위: 앱 시작할 때 실행 (Client 생성)
    - 아래: 앱 종료할 때 실행 (Client 정리)

    async with와 같은 원리 — 열기/닫기를 자동 관리
    """
    # 시작: Client 생성하고 app.state에 저장
    app.state.client = httpx.AsyncClient(timeout=30.0)
    print("httpx Client 생성 완료")

    yield  # 여기서 앱이 실행됨 (요청을 받기 시작)

    # 종료: Client 정리
    await app.state.client.aclose()
    print("httpx Client 정리 완료")


app = FastAPI(lifespan=lifespan)


# --- 모델 정의 ---

class ChatRequest(BaseModel):
    message: str
    model: str = "gpt-4.1-nano"
    max_tokens: int = 200


class ChatResponse(BaseModel):
    reply: str
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# --- 엔드포인트 ---

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    채팅 메시지를 받아서 LLM 응답을 반환

    app.state.client를 사용 — lifespan에서 만든 Client를 재활용
    """

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }

    body = {
        "model": request.model,
        "messages": [{"role": "user", "content": request.message}],
        "max_tokens": request.max_tokens,
    }

    # lifespan에서 만든 Client 사용 (요청마다 새로 만들지 않음)
    response = await app.state.client.post(
        OPENAI_URL, headers=headers, json=body
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"OpenAI API 에러: {response.text}",
        )

    data = response.json()
    usage = data["usage"]

    return ChatResponse(
        reply=data["choices"][0]["message"]["content"],
        model_used=data["model"],
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
    )


@app.get("/health")
async def health():
    """서버 상태 확인 — 프로덕션에서 모니터링용으로 필수"""
    return {"status": "ok"}
