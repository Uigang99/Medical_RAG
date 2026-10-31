# PCED와 동일한 CoT 출력 규칙의 기준선

목적: 기존 TEST 6,545문항에서 PCED-CoT가 No-RAG/Base-RAG보다 나은지를 출력 형식을 통제하여 비교한다. 학습·검색·프롬프트 튜닝은 하지 않는다.

| 조건 | 입력과 생성 |
|---|---|
| No-RAG-CoT | 질문+선택지, 일반 greedy logits |
| Base-RAG-CoT | 질문+선택지+기존 balanced_ce Top-8 원문 청크, 일반 greedy logits |
| PCED-CoT | 완료된 question_only_pced_medcpt_two_modes_v1/rationale 결과 재사용 |

세 조건은 동일한 pristine Llama-3-8B-Instruct, Transformers/BF16/SDPA, chat template, rationale system instruction, 질문→선택지→근거 순서, 8,192 문맥 한도, 512 rationale 토큰 상한을 사용한다. 최종 `Final answer:` 뒤에는 공백 포함 A/B/C/D 단일 토큰 중 가장 높은 점수를 선택한다. 자연 종료 marker가 없으면 EOS 또는 토큰 상한에서 기존 KV에 `\nFinal answer:`를 붙인 뒤 같은 라벨 선택을 수행한다. 이는 세 조건에 동일하게 적용되는 출력 프로토콜이며 무제한 자유 CoT가 아니다.

Base-RAG는 Top-8 전부를 한 입력에서 읽고, PCED는 문서별 expert와 amateur를 이용한다. 이것이 의도한 방법 차이다. No-RAG는 같은 instruction을 사용하되 문서가 없다. 배치 모양은 방법별로 다르므로 BF16 연산의 비트 단위 일치는 주장하지 않는다.

기존 PCED manifest, 소스/입력 hash, 모델 파일 상태, 패키지 버전, 문항 ID, gold, 문서 순서와 priors를 검증한다. 모든 Base-RAG 입력에 전체 생성 여유를 확보하며 초과 시 모델 적재 전 실패한다. 몰래 절단하거나 문항을 제외하지 않는다. 정답은 decoder 입력에 주지 않고 종료 후 집계에서만 사용한다.

## 실행

```bash
bash /home/user/Uiheon/Medical_RAG/scripts/run_question_only_cot_baselines.sh \
  --max-batch-questions 64 \
  --memory-fraction 0.88
```

GPU1에서 모델을 한 번 적재하고 두 기준선을 순차 실행한다. 길이별 배치, KV cache, 완료 문항 cache 압축, 여유 VRAM 기반 배치 수 제한을 사용한다. OOM이면 저장 전 배치만 줄여 재시도하며 문맥/출력 한도는 유지한다. 결과는 문항별 파일 대신 배치 JSONL로 원자적 저장하여 fsync 부담을 줄였다. 같은 명령어는 완료된 배치를 재사용한다. 다른 설정은 새 output-root를 사용해야 한다.

GPU 실행 없이 사전 확인하려면 같은 명령에 `--preflight-only`를 붙인다. 소규모 실행은 `--limit 64`이며 자동으로 별도 smoke64 디렉터리에 저장한다. TEST smoke는 실행 확인용이지 프롬프트/정책 선택용이 아니다.

## 출력과 지표

`results/question_only_cot_baselines_pced_matched_v1/`:

- `manifest.json`, `preflight.json`: frozen 비교 조건과 CPU 검사.
- `{no_rag_cot,base_rag_cot}/shards/*.jsonl`: rationale, 토큰, 최종 답, 종료 유형, 문서 ID.
- `runtime.json`: 조건별 실제 속도와 GPU peak allocated memory.
- `summary.json`, `paired_predictions.jsonl`: 세 조건 대응 비교.

주 지표는 Macro-3 = (MedMCQA 정확도 + MedQA 정확도 + 의료 MMLU 6과목을 합친 정확도)/3이다. 주 비교는 PCED-CoT − Base-RAG-CoT. Micro, 데이터셋별 정확도, C→W/W→C, 데이터셋 층화 paired bootstrap 95% 구간도 기록한다. 전체 문항의 전환 건수는 Micro 통계이며 Macro-3 변화와 혼동하지 않는다.

새 단일-stream CoT의 호환 처리속도는 아직 측정하지 않았다. No-RAG와 Base-RAG 각각 첫 배치부터 별도로 ETA를 보정한다. Base-RAG는 PCED의 개별 expert보다 입력이 길고 생성 길이도 달라 기존 PCED 시간을 단순히 9로 나누어 예상하지 않는다. 아직 측정되지 않은 후속 단계가 있으면 전체 ETA는 calibrating을 유지한다.
