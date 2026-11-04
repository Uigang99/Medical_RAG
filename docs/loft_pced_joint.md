# Temporary Joint Expert를 추가한 PCED 파일럿

## 목적과 범위

동일한 CoT prefix에서 문서에 의해 달라진 예측이 크게 다른 두 Expert를
찾아, 해당 원문 문서 두 개를 함께 읽는 임시 Expert를 기존 PCED 경쟁에 추가한다.
독립 Expert를 제거하거나, 문서를 새로 검색하거나, 답을 되돌리거나,
다른 branch를 생성하지 않는다. PASS 프롬프트나 잔차 점수도 사용하지 않는다.

**제한된 파일럿으로 실행 가능**하다. 기존 Llama-3.1-8B와 LOFT 검색 캐시를
재사용하지만 이 방식의 성능 개선은 아직 검증하지 않았다. 기본 50문항은 이미
검토한 TEST 5개 데이터셋 × 10문항이며, 동작 진단용이다. 이 결과로 threshold를
최적화하거나 최종 일반화 성능을 주장하지 않는다. 독립 validation 검색 캐시는
이번 구현 범위에 포함하지 않았다.

## 입력과 고정 조건

- `results/loft_pced_paper_v1_full500_parallel2`의 검증된 Top-90 원문/순서/검색점수.
- 같은 frozen Llama, 원본 PCED system prompt, 아래 공통 CoT 출력 지시.
- `Based on the documents above, answer the following query. Write brief reasoning before a concise final answer.\nquery: {question}`
- assistant prefix `Reasoning:`, reasoning 최대 64토큰 후 `\nAnswer:`를 강제 추가,
  답 최대 32토큰. EOS 시에도 답 구간으로 전환한다. 두 arm 모두 같은 프로토콜이다.
- 이는 자유로운 종료/완성된 추론을 보장하는 프로토콜이 아니다. 예산 종료와 빈
  reasoning/answer를 별도로 기록한다. 기존 결과와 숫자만 비교하지 않고 PCED
  baseline을 동일한 구현/프로토콜로 새로 생성한다.
- Gold는 생성 이후 EM 평가에서만 읽는다. reasoning 안에 정답 문자열이 있어도
  점수를 주지 않고 답 구간만 기존 Subspan EM으로 평가한다.

## Joint Expert 선택

1. 동일 prefix에서 amateur logit `z0`, 문서별 expert logit `zi`를 얻는다.
2. 문서 효과 크기는 `JSD(softmax(zi), softmax(z0))`로 측정한다.
3. 최종 retrieval prior가 최대 prior의 0.25배 이상이고 효과 JSD가 0.001 nats
   이상인 expert들을 후보로 둔다. 이는 검증되지 않은 초기 기본값이다.
4. 각 후보의 `softmax(zi-z0)`를 만든 뒤 모든 후보쌍의 JSD를 계산한다.
   전체 vocabulary를 쓰되 16쌍씩만 GPU에 비교하여 메모리를 제한한다.
5. 이미 사용하지 않은 쌍 중 JSD가 가장 크고 0.05 nats 이상인 쌍을 선택한다.
   동률은 원래 검색순위가 앞선 쌍이다. 조건을 만족하지 않으면 아무것도 추가하지 않는다.

JSD는 **예측 차이**이지 의미적 모순, 사실성, 상보성의 판정이 아니다.
특히 logit 차이의 softmax는 amateur가 매우 낮게 평가한 희귀 토큰에도 민감하다.
현재는 사용자의 문서 유발 차이 정의를 그대로 구현하며, 유효성은 파일럿에서
선택 쌍/최종 정오 변화/실제 토큰 변경량을 보고 평가해야 한다.

## 동작과 점수

- 기본 warmup 8토큰, 비활성 상태에서 매 8토큰마다 검사한다.
- 최대 2회 생성, 동시에 하나만 유지, 한 번에 최대 32개 sampled position 동안
  사용한다. 새 생성은 reasoning에서만 가능하고 이미 활성화한 Expert는 남은
  수명 동안 답 구간에도 참여할 수 있다. 강제 delimiter는 수명을 소모하지 않는다.
- 문서 순서는 원래 검색순위, 문서 사이 구분자는 두 줄바꿈이다. 질문과 **실제
  선택된 모든 prefix token ID**를 동일하게 제공한다. 두 문서의 KV를 이어 붙이지
  않고 합친 입력을 새로 prefill한다. 이후에는 별도 KV를 재사용한다.
- 독립 Expert들과 amateur는 기존 91-stream batch/KV를 계속 사용한다. Joint는
  한 stream만 추가 연산하며 모든 선택 토큰과 강제 delimiter를 동일하게 받는다.
- 독립 Expert score는 pinned PCED 그대로다. 실제 pinned 구현은 첫 position의
  `get_jsd`가 반환하는 **vocabulary별 elementwise beta**를 고정 사용한다.
  이를 임의로 scalar로 고치지 않는다. Joint beta는 활성화 시점의 amateur와
  계산한 같은 형태의 beta를 해당 수명 동안 고정한다.
- Joint prior 기본값은 **이미 정규화된 두 PCED prior의 기하평균**:
  `r_joint = sqrt(r_a * r_b)`, 즉 log prior를 산술평균한다. raw IP/CE logit의
  평균이 아니며 두 문서를 가졌다는 이유로 prior를 합산하지 않는다.
- Joint도 `score=(1+beta)*z_joint-beta*z0+2.5*log(r_joint)`로 경쟁한다.
  부모 점수와 prior는 재정규화하지 않는다. 정확한 동률은 기존 독립 Expert가 이긴다.
- `--joint-prior arithmetic` 또는 `minimum`은 명시적 추가 비교용 옵션이다.
  기본 실행은 기하평균 한 조건만 사용한다.

## 실행

```bash
bash /home/user/Uiheon/Medical_RAG/scripts/run_loft_pced_joint.sh --dry-run
bash /home/user/Uiheon/Medical_RAG/scripts/run_loft_pced_joint.sh
```

H200 GPU1, 기본 50문항에 `pced`와 `joint` 두 조건을 연속 실행한다. 캐시/원본
결과는 덮어쓰지 않는다. 질문별 atomic 저장으로 동일 명령 재실행 시 완료 문항은
재사용한다. 코드/프롬프트/모델/입력/파라미터가 바뀌면 새 output-root가 필요하다.
정확한 새 경로 dry run은 tokenizer/입력/문서/모델 파일과 cache contract만
확인하며 모델 weight를 로드하거나 GPU generation을 하지 않는다.

새 CoT/Joint의 측정된 처리량은 아직 없다. 기존 즉답 처리시간을 그대로 적용할
수 없으며 각 arm의 첫 5개 새 문항 이후 해당 arm의 ETA를 보정한다. 두 arm의
비용을 알기 전까지 전체 ETA는 unknown이다. Joint prefill과 JSD가 추가되므로
기존 PCED와 동일한 실행시간이라고 가정하면 안 된다.

## 기록과 판단

`manifest.json`, `pced|joint/predictions/<dataset>/*.json`, `summary.json`,
`summary.csv`를 저장한다. 문항별 원문 결과에는 최종 답, reasoning, 토큰별
선택 Expert, 활성 Joint, 동일 prefix에서 독립 PCED만 사용했을 때의 token,
pair 선택/필터/JSD/prior/수명, 실제 latency가 포함된다.

주지표는 **Macro-5 Subspan EM**이다(의료 벤치마크 Macro-3가 아님).
동일 문항의 C→W/W→C, 데이터셋별 정확도, paired bootstrap 구간, 빈 답,
예산 종료, Joint activation/win/실제 token 변경 수와 시간 배율도 보고한다.
동일 prefix에서 다른 token을 골랐다는 것은 정답 개선의 인과적 기여를 뜻하지 않는다.

확장 전 별도 held-out 파일럿 기준 제안: Macro-5 +2%p 이상, W→C>C→W,
latency ≤1.5배, 출력 실패 증가 없음. 50 TEST 진단으로 이 기준을 확정하거나
성능을 입증하지 않는다. 활성화가 없거나, 자주 이겨도 C→W가 커지거나,
latency가 과하면 전체 데이터로 확장하지 않는다. 문서 결합 효과와 단순한 후보
추가 효과를 분리하려면 이후 비용을 맞춘 random-pair 대조군이 필요하다.

## 검증

```bash
cd /home/user/Uiheon/Medical_RAG
.venv_pced/bin/python -m unittest discover -s tests -p test_pced_joint.py -v
```

CPU 임의 초기화 작은 Llama로 Joint-off의 pinned PCED token/winner 동등성,
실제 KV cache/full forward 일치, 공동 prefix와 delimiter, EOS 제외, 수명,
후보 필터와 prior를 검증한다. 이는 H200/BF16 성능이나 정확도 검증을 대신하지 않는다.
