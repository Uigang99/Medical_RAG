# 문장별 참여 판정 PCED 파일럿

현재 가능한 범위는 제한된 개발 파일럿이다. 기존 per-token PASS가 문법적 이어 쓰기와
Answer 경계에 민감했던 문제를 검사한다. 품질 개선을 이미 확인한 방법은 아니다.
학습, 신규 검색, 다른 모델 다운로드, TEST 평가를 하지 않는다.

## 입력과 비교 조건

- 기존 공식 LOFT dev 50문항(5개 dataset × 10), 동일 CE Top90·순위·prior.
- 로컬 Llama-3.1-8B-Instruct 한 개, H200 GPU1만 사용.
- 세 조건 모두 같은 기존 **PASS 없는** 생성 프롬프트, 자연스러운 Reasoning → Answer,
  reasoning 128 / answer 32 토큰, 동일 parser·Subspan EM.
- `pced`: 참여 판정·보정 없음.
- `sentence_exclude`: 별도 판정으로 기권한 expert를 문장 전체에서 제외.
- `sentence_weighted`: 같은 제외 + 판정 확률을 문장 전체의 점수에 반영.

과거 per-token 실험은 생성 프롬프트/판정 방식/점수식이 달라 역사적 참고선일 뿐이다.
새 세 조건은 생성 프롬프트를 통일해서 다시 실행한다. 정답·qrels는 채점/기록에만 쓰고
판정과 생성에는 절대 전달하지 않는다. dev는 이미 사용한 표본이며 미사용 test가 아니다.

## 문장 시작 전 별도 판정

`질문 + 자신의 문서 + 현재 공유 reasoning + 다음 단계(reasoning/answer)`를 같은 Llama에
판정 전용 프롬프트로 준다. 정확한 프롬프트는 `pced_sentence_pass.py` 및 manifest에 저장된다.

- C: 미해결 부분을 진행시킬 구체적인 사실·중간 연결·잘못된 공유 reasoning의 교정 근거가 있다.
- P: 그런 근거를 특정할 수 없다. 주제가 같거나 문장을 이어 쓸 수 있다는 것만으로는 부족하다.
- 완전한 정답이 없어도 부분 근거로 참여할 수 있다.
- `Decision:` 뒤 단일 토큰 ` C` / ` P` logits를 정규화한다.

```
c_i = softmax([z_i(C), z_i(P)])[C]
p_i = 1 - c_i
active_i = c_i > 0.5
base_i(v) = (1 + beta_i(v))*z_i(v) - beta_i(v)*z_amateur(v) + 2.5*log(prior_i)
weighted_i(v) = base_i(v) + 4*log(max(c_i, 1e-6))
```

판정 확률은 C/P 사이 상대 선호이지 검증된 기여 확률이 아니다. 전체 어휘 중 C/P의 확률합과
원래 1위 토큰도 저장해 출력 형식 준수 여부를 분석할 수 있다. tie는 PASS다.

판정과 생성의 KV·프롬프트는 분리한다. C/P나 판정 지시문은 공유 reasoning에 추가하지 않는다.
문장 안에서는 active 집합과 c가 고정되지만, 매 토큰의 승리 expert는 달라질 수 있다.
모든 expert는 기권 여부와 무관하게 공유 토큰을 받으며 다음 문장에서는 전원 재판정한다.

**그림과의 차이:** 그림의 가중합이 아닌 기존 PCED의 expert/token 최대점수 선택을 유지한다.
문장별 판단 효과와 decoder 변경을 섞지 않기 위해서다. 이 가중치는 expert별 상수이므로
동일 expert 내부의 토큰 순위를 직접 바꾸지 않고, expert 선택을 통해 토큰을 바꾼다.

전원 기권하면 해당 문장 동안 모든 문서 expert를 제외한 채 amateur의 일반 토큰을 사용한다.
검색 상위 expert를 몰래 복구하지 않는다. 이것은 문서 근거 선택 성공이 아닌 명시적 fallback이다.
Answer 시작에서도 한 번 새로 판정하고, 짧은 최종 답변이 끝날 때까지 고정한다.

## 문장 경계와 한계

기존 자연 생성문을 읽으며 토큰 끝의 `. ! ?`와 닫는 따옴표를 탐지한다. 흔한 약어·이니셜·숫자
뒤 마침표는 제외한다. 줄바꿈만으로는 재판정하지 않는다. 숫자나 약어로 끝나는 문장은 합쳐질
수 있다. 토큰을 나누거나 이미 생성한 토큰을 되돌리지 않는다. 따라서 완벽한 문장 파서는 아니다.
문장 끝 이후 다음 토큰이 Answer 헤더일지는 미리 알 수 없어 그 직전의 reasoning 판정이 불필요하게
한 번 더 발생할 수 있다. Answer 헤더 인식 직후에는 answer 전용 판정을 한다.
128 토큰 cap에 도달하면 기존처럼 Answer 헤더를 강제하고 이를 기록한다.

## 효율성과 기록

하나의 Llama를 한 번만 로드한다. 생성은 기존 91개 스트림 배치·KV를 유지한다.
판정용 문서+질문 prefix도 최초 한 번 계산하고 길이순 batch로 캐시한다.
매 판정에는 짧은 공유 reasoning/phase suffix만 계산하고 cache를 prefix 길이로 crop한다.
기본 judge batch=16. 비활성 expert도 cache를 유지하므로 제외가 곧 GPU 계산량 감소를 뜻하지 않는다.
두 cache가 공존하므로 preflight에서 KV 예산을 합산하고 모델 로드 전 free VRAM을 확인한다.

각 문항 결과에는 판정 시점의 공유 reasoning, 90개 C/P 확률·logits, active 집합, 모델의 raw top token,
판정 시간, 매 생성 위치의 제안 토큰·선택 expert·원시/보정 점수·동일 prefix에서의 규칙별 선택을 저장한다.
이로써 재참여·잘못된 기권·fallback·정답 토큰 억제·검색 prior 영향을 추적할 수 있다.
`summary.json/csv`: 데이터셋별 및 Micro/Macro-5, C→W/W→C, paired bootstrap, 빈 답변,
budget 종료, 재참여/이탈, 판정 호출 수·시간 및 fallback을 기록한다.

일단 50문항에서 가중 조건이 PCED보다 3문항 이상 추가 정답, 빈 답변 1개 이하를 수치적
진행 기준으로 둔다. 이 기준만으로 일반화나 유의성을 주장하지 않는다. 문서 근거 활용 개선인지
amateur fallback·표기 변경인지 정성 확인 후에만 별도 검증으로 확장한다. 자동 튜닝은 없다.

## 실행

```bash
bash /home/user/Uiheon/Medical_RAG/scripts/run_loft_pced_sentence_pass.sh
```

기본 결과: `/home/user/Uiheon/Medical_RAG/results/loft_llama_sentence_pass_dev_v1_n50/`.
같은 명령은 완료 문항을 재사용하며, 중단된 문항만 다시 시작한다. 계약 변경은 새 output root가 필요하다.
`--dry-run`은 실제 cohort·토크나이저·경로·캐시·문맥만 검사하고 가중치/GPU/결과 파일을 사용하지 않는다.
`--limit-per-dataset 1`은 별도 n5 root의 동작 검사이며 품질 평가는 아니다.

직전 per-token 실험은 200 question-arms에 약 45분이었다. 이번 기본 150 question-arms에는
별도 prefix prefill·문장별 판정이 추가된다. H200 한 장에서 45–120분은 초기 비용 여유 범위일 뿐
실측 ETA가 아니다. 새 조건별 첫 5개 fresh 문항으로 재보정하며 서로 다른 정책의 처리량을 섞지 않는다.
CPU 회귀검사·실제 tokenizer dry-run은 구현 검증이지 H200 실행 또는 판정 품질 검증이 아니다.
