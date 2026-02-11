"""
FastAPI 첫 번째 API 서버

여기서 데코레이터를 자연스럽게 만나게 됨:
- @app.get("/") → "GET 요청이 / 경로로 오면 이 함수를 실행해"
- 데코레이터 = 함수 위에 붙여서 추가 기능을 부여하는 문법
"""

from fastapi import FastAPI

# FastAPI 앱 인스턴스 생성
app = FastAPI()


# 데코레이터: GET /
# "루트 경로(/)에 GET 요청이 오면 이 함수를 실행해"
@app.get("/")
async def root():
    return {"message": "Hello, FastAPI!"}


# GET /hello/{name} — 경로 파라미터
# URL의 일부를 변수로 받음
@app.get("/hello/{name}")
async def hello(name: str):
    return {"message": f"안녕하세요, {name}님!"}


# GET /add?a=1&b=2 — 쿼리 파라미터
# URL 뒤에 ?key=value 형태로 전달
@app.get("/add")
async def add(a: int, b: int):
    return {"result": a + b, "expression": f"{a} + {b} = {a + b}"}
