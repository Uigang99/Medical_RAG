# 출력 지시 복원 파일럿

목적은 새 문서 활용 지시를 유지한 상태에서, 출력 지시 블록을 기존 문장으로 복원하면
첫 토큰 `Final` 선택과 설명 생략이 줄어드는지를 확인하는 것이다.

## 유일한 프롬프트 변경

기존 focused 프롬프트에서 `Give a brief explanation ...`부터 마지막 `Finish ...`까지를
다음 문장 하나로 교체한다. 앞선 `Solve ...`, `The question ...`, `Use ...`는 그대로다.

```text
Explain your reasoning briefly, then finish on a new line with 'Final answer: <option letter>'.
```

이 교체에는 `rather than a summary of the passages`의 제거도 포함된다. 따라서
출력 지시 블록 전체의 대조 실험이지, `then` 한 단어나 문단 간격만의 실험은 아니다.
No-RAG/amateur와 RAG/expert의 기존 역할별 프롬프트 구분도 유지한다.

## 비교와 고정 조건

- 기존 TEST 6,545 중 SHA256(`42:sample_id`) 순서로 256개를 고정 추출한다.
  정답이나 기존 설명 생략 여부로 선별하지 않는다. TEST-informed 원인 진단이며 독립 검증이 아니다.
- 동일 문항의 최초 프롬프트와 focused 프롬프트 결과는 이미 완료된 캐시에서 가져온다.
  No-RAG, Base-RAG, PCED 세 조건의 복원 버전만 새로 생성한다.
- 같은 Llama, Top-8, 질문·선택지·문서 순서, BF16/SDPA, 512토큰 상한과 종료 규칙.
  추론 prefix 강제, 최소 생성 길이, 조기 Final 차단은 추가하지 않는다.
- PCED 점수식과 prior는 유지한다. beta는 기존 공식으로 새 분포에서 다시 계산한다.
- 성능 지표는 Macro-3. 이번 원인 진단 지표는 첫 토큰 Final 비율 및 설명 생략률이다.
  최종 `Final answer:` 앞의 문자열이 비어 있는지를 설명 생략으로 측정하며,
  비어 있지 않다고 해서 설명의 타당성까지 검증된 것은 아니다.
- 강한 형식 복구의 사전 기준: Base-RAG와 PCED 모두 첫 Final 비율이 focused 대비
  20%p 이상 감소하고 paired bootstrap 95% CI 상한이 0 미만이며 설명 생략률도 감소.
  이 기준은 모델 정확도 개선의 기준이 아니며, 실행 후 전체 평가로 자동 확대하지 않는다.

## 실행

```bash
bash /home/user/Uiheon/Medical_RAG/scripts/run_restored_cot_instruction.sh \
  --limit 256 \
  --max-batch-questions 64 \
  --pced-batch-questions 32 \
  --memory-fraction 0.88
```

GPU 1, 모델 한 번 로드, 길이별 배치, KV 재사용, 종료 문항 제거, OOM 배치 축소.
256문항 × 세 조건은 이전 전체 CoT 처리량의 비례 추정과 로딩/검사 비용을 합쳐
약 5–10분 예상이다. 실제 출력 길이에 따라 조건별 ETA를 보정한다.
`--preflight-only`는 동일 계약의 CPU 검사만 수행한다. GPU 모델은 로드하지 않는다.

출력은 `results/restored_cot_instruction_v1_test256/`에 저장한다.
`summary.json`에는 세 프롬프트 버전의 생성 행동, focused 대비 변화와 신뢰구간,
Macro-3와 정답 전환을 기록한다. `shards`에는 실제 생성 텍스트·토큰·beta를 남긴다.
동일 명령어 재실행은 완료 shard를 검증하고 남은 문항부터 재개한다.
기존 코드·결과를 덮어쓰지 않는다. 별도 문항 수에는 별도 출력 디렉터리를 사용한다.
