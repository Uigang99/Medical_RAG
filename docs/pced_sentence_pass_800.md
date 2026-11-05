# Sentence-PASS reasoning v2: 800문항 실행

## 고정한 실험

목적은 Joint와 결합하기 전, 정리된 reasoning 형식에서 문장별 PASS 판단의 효과를 평가하는 것이다.
새 retriever, 학습, Joint, 별도 최종 답변 Llama는 사용하지 않는다.

- LOFT 500문항: HotpotQA / NQ / MuSiQue / Qampari / QuEST 각각 기존 TEST100.
- 의료 300문항: BioASQ / PaperSearchQA / COVID-QA 각각 기존 seeded100.
  새로 랜덤 추출하지 않는다. 의료 문항은 train-origin pilot이다.
- 모든 데이터에 Llama-3.1-8B-Instruct, Top90, gamma=2.5, 기존 BGE/CE prior를 사용한다.
  **과거 의료 CoT의 Qwen·Top10·gamma4 설정은 가져오지 않는다.** 이번 Llama PASS 실험을
  확장하는 것이며, 그 Qwen 결과와는 변인 통제된 비교가 아니다.
- LOFT 검색은 full-corpus CE, 의료 검색은 FAISS Top256→CE Top90의 기존 캐시이다.
  검색·reranking·모델 다운로드는 재실행하지 않는다.
- 세 조건: `pced`, `sentence_exclude`, `sentence_weighted`. 800×3=2,400개 조건별 결과.
- 공통 reasoning v2 prompt, 최대192토큰, 자연 `Answer:` 전환, Answer32토큰.
  `Answer:` 없이 한도에 걸리면 빈 답변/오답으로 기록한다. 해당 문항은 분모에서 빼지 않는다.
- PASS는 문장 시작 전 별도 C/P 판정이다. c>0.5인 expert만 참여하고 weighted는
  기존 PCED 점수에 4*log(c)를 추가한다. 문장 안에서는 gate 고정, 다음 문장에서 전원 재판정.
  전원 PASS 시 기존처럼 amateur가 해당 구간을 생성하며 fallback을 별도로 기록한다.

800문항은 이미 여러 번 본 평가 집합이다. frozen exploratory evaluation이지 미사용 최종
시험이 아니다. 실행 중 프롬프트·threshold·gamma·길이를 자동으로 선택하거나 수정하지 않는다.
기술적으로 실행 가능한 구성이나, DEV50에서 개선을 확인한 후 확장하는 순서를 건너뛴 사용자의
요청에 따른 전체 비교이므로 개선 가능성이 검증됐다는 뜻은 아니다.

## 계산 재사용: 방법은 그대로, 같은 입력의 중복 forward만 제거

문항 하나에서 weighted → exclude → pced 순서로 계산한다. 이미 완료된 조건은 건너뛴다.

1. 첫 조건은 기존 91개(amateur+90 experts) 스트림을 같은 배치·dtype·attention으로 계산한다.
2. 다른 조건의 counterfactual 선택이 첫 조건과 같은 **토큰**이면 그 위치의 forward와 결과를
   재사용한다. expert가 달라도 공유 토큰이 같으면 다음 입력과 KV는 동일하다.
3. 선택 토큰이 처음 달라지는 위치에서, 최초 경로의 **그 위치 이전까지만** KV를 복사한다.
   이후에는 별도 cache에서 해당 조건을 정상적으로 계산한다. 두 조건의 토큰을 섞지 않는다.
4. judge의 문서+질문 prefix는 문항 내에서 한 번 만든다. `공유 텍스트 + phase`가 정확히
   같은 판정만 재사용한다. 텍스트가 달라지면 새로운 판단을 한다.

완성된 답변이 비슷하다는 이유로 재사용하지 않는다. 문장 유사도, gold, 정답률을 이용한 재사용도
없다. 첫 분기 뒤 재합류를 추정하지 않는다. 선택 expert/점수/토큰과 동일 prefix 반사실 기록은
각 조건별로 보존한다. `pced` 기록에는 gate 없이 계산한 상태를 기록한다.

동시에 여러 질문을 억지로 묶어 패딩과 수치 경로를 바꾸지 않는다. 이미 91개 expert를 배치로
계산하며, 추가 VRAM은 문서/judge prefix와 분기 KV 보존에 사용한다. judge batch16도 기존과
동일하다. 단순히 VRAM을 모두 채우는 것을 목표로 하지 않는다.

첫 실제 문항의 짧은 3-step GPU 계산으로 기존 cache와 분기 cache의 다음 토큰 및 expert
점수가 정확히 같은지 검사한다. 실패하면 전체 생성 전에 중단한다. CPU에서는 완전 재사용,
부분 분기, 같은 토큰/다른 expert, 전원 PASS와 재참여, 원본 KV 불변성을 검사한다.
이 검사는 알고리즘 동작 검증이지 최종 생성 품질 검증이 아니다.

## 실행과 기록

```bash
bash /home/user/Uiheon/Medical_RAG/scripts/run_pced_sentence_pass_800.sh
```

GPU1만 사용하며, Llama는 한 번 로드한다. 코드 제공 단계에서는 전체 GPU 실행을 시작하지 않는다.
`--dry-run`을 붙이면 가중치/GPU/결과 쓰기 없이 실제 800문항의 소스·캐시·tokenizer·문맥·재개 계약을 검사한다.

기본 출력:
`/home/user/Uiheon/Medical_RAG/results/pced_sentence_pass_reasoning_v2_800_v1/`

- `manifest.json`: 전체 계약, 800문항 ID, 소스 및 코드 hash, prompts, 모델, score, 형식/중단 정책.
- `runtime_safety.json`: 실제 모델의 cache fork 동등성 검사.
- `{arm}/predictions/{dataset}/*.json`: 원문 답변·reasoning, 90개 판정/토큰 점수, 실제 계산/재사용 수.
- `summary.json/csv`: dataset별 정확도, Micro/Macro8, LOFT Macro5/의료 Macro3, C→W/W→C,
  paired bootstrap, 미완성/빈 답변, 재참여 및 재사용 수.

문항-조건별 atomic 저장이다. 중단 시 같은 명령으로 완료 조건을 건너뛰며, 미완료 조건만 다시
실행한다. 이전 프로토콜의 결과는 덮어쓰지 않는다. 설정/코드 계약이 다르면 재개를 거부한다.
`--independent-arms --output-root <새 경로>`는 재사용 없는 진단용 비교이며 기본 실행에 필요하지 않다.

3조건을 함께 비교할 때만 이번 재사용 최적화의 이득을 크게 얻는다. 조건별 `seconds`는 실제
발생 비용을 첫 실행 조건에 배분하므로 조건 간 독립 추론시간 비교에 쓰면 안 된다.
전체 경과 시간과 실제/reused forward 수를 함께 확인한다.

## 자원과 예상 시간

전체 800문항 사전 검사: 최대 generation context+budget 2,009토큰, 두 생성 KV와 judge KV를
합친 보수적 최대 약53.7GiB. 모델·임시 메모리 여유를 포함한 시작 전 free VRAM 요구량은
약93.1GiB이다. GPU1 H200 NVL의 확인 당시 free VRAM은 약139.8GiB로 범위 내였다.
summary는 2,400개의 대형 trace를 모두 RAM에 올리지 않고 한 문항씩 집계한다.

직전 DEV50×3조건은 25분49초, 단순800 환산은 약6.9시간이었다(이전128토큰 protocol).
이전 weighted 경로에서 다른 조건과 동일했던 prefix는 pced 2,676/3,583 위치,
exclude 2,937/3,583 위치였다. 이는 **이전 경로에서 재사용할 수 있었던 위치 수**이며 새 출력의
속도 향상 보장이 아니다. 기존 전체-동일 경로 수는 각각31/50,35/50이다.

이번 계획 범위는 GPU1, 세 조건, judge batch16에서 약3–7시간이다. 새 prompt가 긴 reasoning을
많이 생성하면 이보다 오래 걸릴 수 있다. 최초5개 fresh 문항의 계산 시간과 입력 길이에 따른
비용 추정으로 미실행 dataset의 ETA를 보정하고, dataset 진입 후에는 해당 dataset 실측으로
갱신한다. 개발 결과 개선이 아직 확인되지 않았으므로 추가 조건/모델/전체 corpus로 자동 확장하지 않는다.
