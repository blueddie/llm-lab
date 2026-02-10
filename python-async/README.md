# Phase 1: Python Async/Await

## 이게 뭐고 왜 쓰는지

LLM API 호출처럼 **대기 시간이 긴 작업**(I/O 바운드)을 동시에 처리하기 위한 Python의 비동기 프로그래밍 방식.
동기 방식으로 10개의 API를 호출하면 30초 걸릴 것을, 비동기로 3초에 처리할 수 있다.

## 핵심 개념

### 1. I/O 바운드 vs CPU 바운드
- **I/O 바운드**: API 호출, DB 쿼리, 파일 읽기 → 대기 시간이 대부분 → **async가 효과적**
- **CPU 바운드**: 데이터 계산, 이미지 처리 → 연산 시간이 대부분 → async 효과 없음

### 2. async/await는 세트
- `async def`: "이 함수는 비동기가 될 수 있어"라고 선언
- `await`: "여기서 기다려야 하니까 다른 작업에게 양보해"
- **await 없이 async def만 쓰면 동기 코드와 동일** (03_async_wrong.py)

### 3. async 안에서는 비동기 라이브러리를 써야 함
| 동기 (쓰면 안 됨) | 비동기 (써야 함) |
|---|---|
| `time.sleep()` | `await asyncio.sleep()` |
| `requests` | `httpx` (AsyncClient) 또는 `aiohttp` |

### 4. Threading vs Async
| | Threading | Async |
|---|---|---|
| 전환 주체 | OS가 결정 | 개발자가 await로 명시 |
| 쓰레드 수 | 작업당 1개 | 싱글 쓰레드 |
| 메모리 | 많이 씀 | 적게 씀 |
| 동시 처리 수 | 수십~수백 | 수천 가능 |

## 동시 실행 패턴

### asyncio.gather() — 묶어서 한번에
```python
results = await asyncio.gather(task1, task2, task3)
```
- 전부 끝날 때까지 기다림, 결과를 리스트로 반환
- 용도: "이 작업들 전부 끝나면 다음으로"

### asyncio.create_task() — 예약하고 나중에 수집
```python
task1 = asyncio.create_task(call_llm(1))  # 즉시 실행 시작
# ... 다른 작업 가능 ...
result1 = await task1  # 필요할 때 결과 수집
```
- 실행 시점과 결과 수집 시점을 분리 가능
- 용도: "작업 돌려놓고 그 사이에 다른 것도 하고 싶다"

## 에러 처리

### return_exceptions=True
```python
results = await asyncio.gather(task1, task2, task3, return_exceptions=True)
for result in results:
    if isinstance(result, Exception):
        # 실패 처리
    else:
        # 성공 처리
```
- 없으면: 하나 실패 시 성공한 결과도 전부 날아감
- 있으면: 에러도 결과로 받아서 개별 처리 가능

### Retry + Exponential Backoff
```python
for attempt in range(max_retries):
    try:
        return await call_llm(user_id)
    except Exception:
        wait = 2 ** attempt  # 1초 → 2초 → 4초
        await asyncio.sleep(wait)
```
- 실패 시 간격을 점점 늘려가며 재시도
- Rate limit 대응에 필수

## 파일 구조

| 파일 | 내용 |
|------|------|
| `sync_baseline.py` | 동기 방식 — 순차 처리 (6초) |
| `async_baseline.py` | 비동기 방식 — 동시 처리 (2초) |
| `async_blocking_pitfall.py` | await 없이 async 쓰면 생기는 문제 (6초) |
| `concurrent_patterns.py` | gather vs create_task 비교 |
| `error_handling.py` | return_exceptions=True 패턴 |
| `retry_with_backoff.py` | 재시도 + exponential backoff |

## 실무 연결

- **FastAPI**가 async 기반인 이유: 수천 개의 동시 요청을 싱글 쓰레드로 처리 가능
- LLM API 호출 시 `requests` 대신 `httpx.AsyncClient` 사용
- 여러 LLM을 동시에 호출해서 결과 비교(A/B 테스트)할 때 gather 사용
- 프로덕션에서는 retry + exponential backoff 필수

## 다음 단계

- [ ] httpx로 실제 HTTP 비동기 호출 실습
- [ ] asyncio.Semaphore — 동시 요청 수 제한 (rate limit 대응)
- [ ] async context manager (async with)
- [ ] 실제 LLM API를 async로 호출하기
