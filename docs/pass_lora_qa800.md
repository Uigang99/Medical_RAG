# 학습된 PASS의 LOFT500 + Medical300 QA 평가

## 목적과 고정 조건

판정용 teacher forcing AUROC가 실제 자유 생성의 QA 개선으로 이어지는지 평가한다.
추가 학습, 재검색, test 기반 임계값 선택은 하지 않는다. 기존 결과와 프롬프트가
다르므로 아래 여섯 조건을 모두 새로 실행한다.

- 동일한 기존 LOFT TEST 5종 각 100문항 및 의료 3종 각 100문항.
- 의료는 BioASQ 11 factoid / PaperSearchQA / COVID-QA이며, 의료 객관식 6,545문항이 아니다.
- 기존 캐시의 CE 순위 Top-16, 순서·문서·paper prior 고정. LOFT와 의료의 기존 검색
  절차는 서로 다르며 검색 계약을 manifest에 각각 보존한다.
- Llama-3.1-8B-Instruct, `pass_lora_v1/best` (validation 선택 step 800).
- 학습에 사용한 `chain_v5`와 두 few-shot 예시를 변경하지 않는다. 문서에 없는 연결은
  모델 지식으로 보완할 수 있게 하는 기존 프롬프트다. `<pass>` 지시를 새로 추가하지 않는다.
- reasoning 최대 256토큰, frozen base reader 최대 48토큰, greedy BF16.
- 모든 조건에서 reader는 **질문 + reasoning만** 받는다. 문서를 다시 주지 않는다.
- 첫 reasoning 위치의 scalar mean JSD beta를 고정하는 프로젝트 기존 CoT-PCED
  구현을 사용한다. 논문 원본 구현의 token-wise JSD와 동일하다고 주장하지 않는다.
- gamma=2.5, PASS penalty 계수=1.0. 이번 QA 결과로 재선택하지 않는다.
- 채점은 기존 Subspan EM, final answer 첫 비어 있지 않은 줄에만 적용한다.
  reasoning의 정답 문자열은 채점하지 않는다. 잘린 reasoning도 미완성으로 표시하고
  reader를 실행하며 평가 분모에서 제외하지 않는다.

## 여섯 조건

| arm | reasoning 생성 | 역할 |
|---|---|---|
| `no_rag` | 문서 없는 frozen base | 내재지식 기준선 |
| `all` | 모든 Top-16을 함께 읽는 frozen base | ALL CoT 기준선 |
| `pced` | frozen base 문서별 expert + frozen amateur | LoRA 없는 PCED |
| `pced_lora` | LoRA 문서별 expert + frozen amateur, 기권 미사용 | 일반 출력 분포의 LoRA 변화 통제 |
| `pass_pced` | 위와 동일, p(PASS) >= 0.7이면 기권 | validation 균형 기준, 주 실험 |
| `pass_pced_safe` | 위와 동일, p(PASS) >= 0.95이면 기권 | validation 참여 우선 기준, 사전 지정 민감도 비교 |

0.7과 0.95는 `judgment_test.json`에 기록된 **validation에서 선택한** 값이다.
이번 800문항에서 성능이 높은 임계값을 새 최적값으로 선언하지 않는다.

매 토큰 위치에서 모든 문서 expert가 다시 PASS 확률을 계산한다.

1. 참여 0개: 그 위치에서 PASS 감점을 제거하고 전체 expert의 ungated PCED로 복귀한다.
   여기서 전체 expert는 **LoRA 적용 expert**다. frozen PCED의 별도 생성 경로로 전환하는 것이 아니다.
2. 참여 1개: 해당 expert의 일반 어휘 중 최대 raw logit 토큰을 선택한다.
3. 참여 2개 이상: 참여 expert에 한해서 아래 점수의 expert-token 최댓값을 선택한다.

```
score_i(v) = (1 + beta) * z_i(v) - beta * z_amateur(v)
             + 2.5 * log(prior_i) + log(1 - p_i(PASS))
```

`--fallback amateur`도 구현했지만 기본 실험에는 포함하지 않는다. 사용할 경우 별도
출력 계약으로 실행한다. Combined Expert는 이번 비교에 넣지 않는다. 기권 효과와 문서
결합 효과를 한꺼번에 바꾸지 않기 위해서다.

기권 expert도 **선택된 일반 토큰을 계속 읽고 KV를 갱신**한다. 다음 위치에 재참여할 수
있다. `<pass>`는 공통 생성문에 들어가지 않는다. 따라서 기권으로 forward 계산이 줄지는 않는다.
amateur와 final reader에는 LoRA를 적용하지 않는다.

PASS 확률은 학습·판정 평가 코드와 같은 별도 출력 행의 V+1 분모를 그대로 사용한다.
원래 reserved-token의 LM-head 행은 그 분모에는 남기고, 일반 토큰 선택에서는 금지한다.
분모를 임의로 바꾸면 기존 0.7/0.95 기준과 의미가 달라진다.

## 실행

실시간 터미널 화면을 유지하면서 SSH 종료에도 실행을 보존하려면 아래 명령을 사용한다.
동일 명령을 다시 실행하면 살아 있는 작업에 연결하고, 완료 상태에서는 완료를 표시한다.
Ctrl+C는 표시 화면만 닫으며 평가를 중단하지 않는다. 오류로 끝난 작업은 같은 명령으로
문항별 완료 기록부터 재개한다. 계산 코드·실험 계약은 변경하지 않는다.

```bash
/home/user/Uiheon/.venv_vllm/bin/python /home/user/Uiheon/Medical_RAG/scripts/run_pass_lora_qa800_live.py
```

진행 화면은 기존 `status.json`을 읽어 전체/현재 단계 두 줄을 표시한다. 계산 로그는 결과
폴더의 `live.log`, 실행/종료 코드는 `live_job.json`에 남는다. 이 방식에는 `nohup`, `&`,
`tail -f`를 별도로 붙이지 않는다. 전체 실행시간은 원래 평가와 동일하며 조건별 최초
2문항 실행 후 ETA가 나온다.

전체 데이터 CPU 계약 검사 (GPU 모델 로드 없음):

```bash
bash /home/user/Uiheon/Medical_RAG/scripts/run_pass_lora_qa800.sh --phase preflight
```

두 실제 질문, 조건별 4토큰만 생성하는 구현 smoke:

```bash
bash /home/user/Uiheon/Medical_RAG/scripts/run_pass_lora_qa800.sh --phase smoke --per-dataset 1 \
  --output-root /home/user/Uiheon/Medical_RAG/results/pass_pced_chain_v5_800_smoke_final
```

본 실행은 **800문항 × 6조건**이다. reasoning과 reader를 각각 생성한다.

```bash
bash /home/user/Uiheon/Medical_RAG/scripts/run_pass_lora_qa800.sh --phase run
```

SSH 연결 종료와 분리해 실행하려면:

```bash
nohup bash /home/user/Uiheon/Medical_RAG/scripts/run_pass_lora_qa800.sh --phase run > /home/user/Uiheon/Medical_RAG/results/pass_pced_chain_v5_800_v1.log 2>&1 < /dev/null &
tail -f /home/user/Uiheon/Medical_RAG/results/pass_pced_chain_v5_800_v1.log
```

중단 후 같은 명령으로 문항별 완료 기록을 재사용한다. reasoning만 완료됐으면 reader부터
재개한다. 모델·코드·문항·프롬프트·임계값·batch 등을 바꾸면 새 출력 폴더가 필요하다.
완료 결과 재집계는 같은 명령의 `--phase summarize`로 수행한다.

## 자원과 검증

GPU 1만 사용한다. 하나의 base model에 LoRA를 활성/비활성화하며, 2문항의 expert들을
배치 처리한다. frozen amateur는 별도 cache에서 계산한다. vLLM/torch.compile은 사용하지
않으며 큰 모델 다운로드나 CUDA 컴파일을 하지 않는다.

실행 전 모델/체크포인트/토크나이저/검색 캐시/입력 길이/디스크/가용 RAM·VRAM을 검사한다.
각 batch 전에 가용 메모리와 디스크를 다시 확인한다. 외부 작업의 갑작스러운 자원 소비까지
막는 전용 자원 예약은 아니므로, OOM이 절대로 발생하지 않는다고 보장하지는 않는다.

CPU 전체 계약 검사에서 최대 prompt는 No-RAG 593 / ALL 8,097 / 단일 expert 1,761토큰이었다.
배치 2의 보수적 KV 추정은 8.4 GiB다. 실제 모델 2문항·4토큰 smoke의 최대 allocated VRAM은
약 24.7 GiB였다. 이는 전체 입력에서의 최대 VRAM 실측값이 아니다.

전체 trace의 모든 출력을 토큰 상한까지 생성했을 때 비압축 상한은 약 129.5 GiB이다.
시작 시 이 값에 여유 공간 30 GiB를 더해 디스크를 검사한다. 실제 짧은 출력과 압축을 사용하면
더 작아진다. 전체 vocab logit 배열을 매 위치 그대로 저장하는 TB급 방식을 쓰지 않는다.

4토큰 smoke만으로 전체 소요 시간을 확정하지 않는다. 시작 시 각 조건의 첫 2문항을 실제
완료하고, reader와 trace 저장까지 포함한 조건별 실측 속도로 ETA를 계산한다. 이후 갱신한다.

검증: gating/재참여/분모/trace 정확성, tiny Llama causal cache, batch/단독 생성,
frozen amateur, reader+결과 저장+재개+paired 집계의 CPU 테스트 8개; 실제 checkpoint의
base/adapted/reader 4토큰 smoke. Batch parity는 검증 표본의 결과이지 모든 GPU batch 형태의
bitwise 일치 보장은 아니다. 실제 KV 재사용과 전체 접두부 재계산의 PASS 확률 최대 차이는
약 0.0044였으며, BF16 연산 형태 차이가 존재한다.

## 산출물과 토큰 분석

기본 경로: `/home/user/Uiheon/Medical_RAG/results/pass_pced_chain_v5_800_v1/`

- `manifest.json`: 평가 계약, 입력·코드·모델/LoRA 식별정보, 임계값 선택 출처.
- `runtime_preflight.json`: 자원/길이 추정과 실행 시 commit.
- `inputs/<dataset>/<qid-sha>.json`: 질문, 답 alias, 문서 전문/ID/순위/prior/qrels.
  gold는 사후 채점과 분석에만 사용하며 생성 함수에 넣지 않는다.
- `reasoning/<arm>/<dataset>/...json`: raw reasoning, token ID, 종료 이유, 미완성·회피 표식.
- `predictions/<arm>/<dataset>/...json`: 위 정보와 reader 출력, 최종 답, Subspan EM.
- `traces/<arm>/<dataset>/...npz`: reasoning 매 위치의 모든 expert 기록.
- `answer_traces/...npz`: frozen reader의 동일 형식 기록.
- `summary.json`, `summary.csv`, `per_question.jsonl`: 8개 데이터셋별 및 LOFT/의료/전체 집계,
  대응 C→W/W→C, paired bootstrap 차이 구간, gate·재참여·종료 통계.
- `status.json`: 전체 및 현재 단계 진행률/ETA.

각 trace에는 모든 expert의 raw top-8, PCED top-8, 선택 토큰에 대한 모든 expert의 정확한
FP32 logit/rank, 전체 어휘 logsumexp, PASS logit/확률, active mask, beta/prior,
선택 expert/token, PASS·prior·contrast 제거 시의 동일 접두부 반사실 선택이 들어간다.
후자는 이후 토큰까지 달라지는 자유 생성의 대체 결과가 아니다.

또한 매 위치·모든 expert의 마지막 hidden state를 BF16 bit pattern으로 보존한다.
동결된 LM head를 곱하면 필요할 때 **전체 어휘 logit**을 재구성할 수 있다. CPU/GPU와
GEMM 형태에 따라 작은 차이가 날 수 있으므로, 실제 선택 분석의 원본값은 직접 저장한
후보 logit이다. 선택·반사실·각 expert top 후보의 합집합을 모두 직접 저장한다.

첫 HotpotQA 문항의 10번째 위치를 읽으려면:

```bash
USE_TF=0 USE_FLAX=0 /home/user/Uiheon/.venv_vllm/bin/python /home/user/Uiheon/Medical_RAG/scripts/inspect_pass_lora_qa_trace.py \
  --run-root /home/user/Uiheon/Medical_RAG/results/pass_pced_chain_v5_800_v1 \
  --arm pass_pced --dataset hotpotqa --step 9
```

특정 문항은 `--qid '<실제 qid>'`로 지정한다. 전체 어휘가 필요하면
`--full-vocab-out /tmp/pass_one_step_logits.npz`를 추가한다. CPU에서 LM head만 읽으며
LoRA 전체 추론이나 GPU 재실행은 하지 않는다. 기존 파일을 덮어쓰지 않는다.

## 판단 기준과 한계

주 비교는 PASS-PCED 대 LoRA-only PCED이며 연구 목표 비교는 PASS-PCED 대 ALL CoT다.
실용적 개선 기준은 각각 +2%p 이상으로 사전 기록하고, 대응 신뢰구간을 반드시 함께 본다.
판정 AUROC 상승이나 expert 교체 증가만으로 성공이라고 해석하지 않는다.
0.95 조건은 사전 지정 민감도 분석이며 결과를 본 뒤 선택한 최적 방법이 아니다.

이 800문항은 이전 연구 과정에서 이미 살펴본 평가 집합이고 의료 표본은 train-origin이다.
새로운 미접촉 test의 확증 결과가 아니라 탐색적 적용 결과로 보고한다. gate는 gold prefix가
아닌 자기 생성 접두부를 보게 되므로 학습 시 성능이 그대로 유지된다는 보장은 없다.
PASS 때문에 정답률이 떨어지거나 LoRA-only 대비 이득이 없으면 이 평가셋에 맞춰 반복
튜닝하지 않고, 실패 위치와 다른 개발 데이터에서 개선 가능성을 먼저 판단한다.
