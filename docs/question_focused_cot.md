# 짧은 질문 중심 프롬프트: CoT 3조건 재평가

## 목적과 범위

질문은 해결할 과제와 주어진 사실, 검색 문서는 참고 지식이라는 역할을 짧게 구분한다.
기존 TEST 오류 분석에서 나온 한 가지 고정 수정안이다. TEST 6,545문항 재평가는
진단 비교이며 독립적인 최종 검증이나 프롬프트 탐색으로 해석하지 않는다.
직접답변 3조건의 코드와 결과는 변경하거나 다시 실행하지 않는다.

## 정확한 system 프롬프트

No-RAG와 PCED amateur:

```text
Solve the question as stated.

The question defines the task and the given facts.

Use the given facts together with your knowledge. Give a brief explanation of why your answer follows.

Finish on a new line with 'Final answer: <option letter>'.
```

Base-RAG와 PCED의 문서 expert 8개:

```text
Solve the question as stated.

The question defines the task and the given facts. Retrieved passages provide reference knowledge; they are not an extension of the question.

Use relevant information from the passages together with your knowledge. Give a brief explanation of why your answer follows, rather than a summary of the passages.

Finish on a new line with 'Final answer: <option letter>'.
```

## 유지하는 조건

- 같은 동결 Llama-3-8B-Instruct, BF16/SDPA, 질문·선택지·문서 텍스트와 순서.
- 전체 DB / 질문 원문 검색 / balanced_ce Top-8. 재검색·학습 없음.
- No-RAG는 문서 없음, Base-RAG는 8문서 통합, PCED는 amateur 1개와 단일 문서 expert 8개.
- 동일한 기존 greedy 생성, 최대 rationale 512토큰, EOS/예산 종료 시 `Final answer:`를 붙이고
  A/B/C/D 중 하나를 고르는 규칙. 정상 marker도 같은 라벨 선택을 사용한다.
- PCED 점수식, gamma=2.5, MedCPT prior, 공통 생성 토큰을 9개 KV stream에 반영하는 동작 유지.
  beta는 기존 규칙대로 새 초기 분포에서 재계산한다. 과거 beta를 재사용하지 않는다.
  따라서 PCED의 변화는 프롬프트와 그에 따른 beta 변화가 함께 반영된 결과다.
- 원본 생성 코드를 상속하고 prefill의 system 문자열만 교체한다. 원문에 같은 문자열이
  있어도 첫 system 부분만 바뀐다. 직접답변 mode는 새 진입점에서 허용하지 않는다.

## 실행

```bash
bash /home/user/Uiheon/Medical_RAG/scripts/run_question_focused_cot.sh \
  --max-batch-questions 64 \
  --pced-batch-questions 32 \
  --memory-fraction 0.88
```

GPU 1에서 No-RAG → Base-RAG → PCED를 순차 실행한다. 모델은 한 번만 올린다.
길이별 묶음, 질문 병렬 배치, KV cache 재사용, 종료 문항 제거, OOM 배치 축소를 사용한다.
기존 결과와 별도 `results/question_focused_cot_test6545_v1/`에 저장한다.
동일 명령어 재실행 시 완료 shard를 검증하고 남은 문항만 생성한다.
출력마다 manifest hash가 기록되며 다른 프롬프트나 설정으로 기존 결과를 덮어쓰지 않는다.

CPU 사전검사만 실행하려면 동일 명령어에 `--preflight-only`를 추가한다.
기존 입력·소스·체크포인트 계약, 6,545문항 대응, 새 prompt 길이를 검사하며 모델은 로드하지 않는다.
`--limit 64`는 별도 smoke 경로를 사용하지만 실행하려면 GPU가 필요하다.

H200의 기존 동일 생성 방식 측정치는 No-RAG 약 7.3분, Base-RAG 32.0분,
PCED 61.6분이었다. 전체 예상은 **약 1.5–2.5시간**으로 잡되 새 프롬프트의 출력 길이와
서버 부하에 따라 달라진다. 각 조건의 실제 새 처리량으로 ETA를 다시 계산한다.

## 출력과 해석

- `summary.json`: 새 3조건의 Macro-3, micro, 데이터셋별 정확도, 종료 유형, 생성 길이;
  조건 사이와 각 조건의 구/신 프롬프트 사이 C→W/W→C 및 paired bootstrap 구간.
- Macro-3는 MedMCQA, MedQA, 6개 의료 MMLU 과목을 합친 정확도 셋의 평균이다.
  MMLU 과목별 정확도를 각각 동일 가중 평균하는 지표가 아니다.
- 주 비교는 새 PCED minus 새 Base-RAG의 Macro-3. 구/신 변화는 추가 진단이다.
- `paired_predictions.jsonl`: 문항별 정답과 구/신 3조건 답변.
- 각 조건의 `shards/*.jsonl`: rationale 원문·토큰, 종료 사유, PCED beta·expert 선택 기록.
- `runtime.json`, `preflight.json`, `manifest.json`: 처리량·사전검사·불변 실험 설정.

같은 시스템 안에서 프롬프트만 바뀌는 비교다. No-RAG는 문서 관련 문구가 없고 PCED
amateur에도 이를 적용하므로, 새 3조건의 모든 stream이 문자 그대로 같은 system은 아니다.
직접답변과 CoT의 과거 backend 차이는 이번 수정으로 해소되지 않으며, 직접답변 결과를
추가로 제시할 때 그 차이를 명시해야 한다. 여기서는 새 CoT 3조건을 통제 비교한다.
