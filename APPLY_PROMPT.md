# Yuseong-Cheon/PIVOT 에 반영하는 프롬프트

Yuseong-Cheon/PIVOT 은 LEE9396/PIVOT 의 **fork** 라 같은 네트워크입니다.

## 어느 PC 에서 무엇을 하나

| PC | 하는 일 | 로봇 명령 |
|---|---|---|
| **로봇 PC** (RB5·AFT200·D456 연결) | 아래 프롬프트로 브랜치 적용 → 보정·타어 → `launch_experiment.sh` 로 **실물 세션** | 있음 |
| 아무 PC (개발용) | `./setup/quickstart_sim.sh` 로 통합 UI 리허설 (모의 장비) | 없음 |

**아래 프롬프트는 로봇 PC 의 Yuseong-Cheon/PIVOT 체크아웃(`~/Desktop/PIVOT`)에서**
Claude Code 에 그대로 붙여넣습니다. 개발 PC 에서 미리 해 봐도 되지만, 보정 파일
(`calibration/*.json`)과 타어는 **로봇 PC 에서 새로** 만들어야 하고 절대 복사하지
않습니다.

---

## 방법 A — 브랜치를 그대로 가져오기 (권장, 5분)

```
LEE9396/PIVOT 의 torque-only-given-mass 브랜치를 이 저장소에 가져와줘.
session_20260904_1736 의 실패(총질량 -27 %, 파지 오프셋 상자 railing, 힘 채널
58.7 N 오프셋)를 고치는 변경이고, 부모가 지금 HEAD(2dd7611)라 충돌이 없어야 해.

  git remote add lee https://github.com/LEE9396/PIVOT.git 2>/dev/null || true
  git fetch lee torque-only-given-mass
  git diff --stat HEAD lee/torque-only-given-mass     # my_work 5개 + tools/preflight.py + setup/experiment.conf.example + 문서
  git cherry-pick 2dd7611..lee/torque-only-given-mass

충돌이 나면 멈추고 어떤 파일인지 알려줘 — 이 저장소가 그 뒤로 더 나갔다는 뜻이니까.

가져온 뒤 다음을 순서대로 돌리고 결과를 보여줘.

  1) 코드 검증
     robot_learning/scripts/run_drake_env.sh python -m unittest discover -s tools -p 'test_*.py'
     robot_learning/scripts/run_drake_env.sh python -c "import sys; sys.path.insert(0,'my_work'); import hardware_real as h; h.self_check()"
     robot_learning/scripts/run_drake_env.sh python my_work/pivot_ui.py --dry-run --sessions /tmp/pivot-dry
     -> 32 tests OK, 13/13, 단계 기계 OK 여야 함

  2) 설정
     setup/experiment.conf 에 추가/변경:
       TOTAL_MASS_KG=<저울 kg, 힌지 포함>     # 램프 0.571
       GRASP_SIGMA_MM=15
       GRASP_MU_MM=                          # 반드시 비움 (measured 에서 이중 계산 금지)
       GRASP_FRAME=measured                  # 유지 확인
       TARE_MODE=gravity                     # 유지 확인
     그리고 robot_learning/scripts/run_drake_env.sh python tools/preflight.py --conf setup/experiment.conf 를
     돌려서 새 행 4개(토크 기준점 / 저울 총질량 / 파지 사전평균 / 파지 불확실성)가 OK 인지 보여줘.

  3) 문서 읽기
     SESSION_20260904_ROOT_CAUSE.md  (왜 실패했나 — 숫자)
     TORQUE_ONLY.md                  (무엇을 바꿨나, 남은 문제)
     TEAMMATE_CHECKLIST.md §6-0      (실험 전 순서)
```

PR 로 받고 싶으면 (fork 네트워크라 upstream → fork 방향 PR 이 됩니다):

```bash
gh pr create --repo Yuseong-Cheon/PIVOT --base real-experiment-ready \
  --head LEE9396:torque-only-given-mass \
  --title "session_20260904_1736 재발 방지: 저울 총질량 + 토크 전용 + preflight 검사" \
  --body-file APPLY_PROMPT.md
```

---

## 방법 B — 직접 적용 (브랜치를 못 가져올 때)

```
이 저장소(PIVOT)의 밀도 추정 파이프라인을 "저울 총질량을 받고 토크 3축만 쓰는"
방식으로 바꾸고, session_20260904_1736 을 재발시키는 설정을 preflight 에서 막아줘.

## 왜

density_id_drake.regressor 의 힘 행 force_rows = FORCE_SIGN * G_ACC * np.outer(g_hat, VOLUMES)
는 세 행이 전부 VOLUMES 의 상수배라 rank 1 이고, 중력 방향을 늘려도 1 이야. 힘 채널이
밀도에 대해 아는 건 총질량 Σ V_i ρ_i 하나뿐이고, 그건 저울이 훨씬 정확히 줘. 반면
실물 힘 채널에는 세션에서 58.7 N 오프셋(램프 5.60 N 의 10 배)이 실렸는데
SIGMA_F = 0.10 N 로 백색화해서 500 배 과신하고 있었어.

## my_work/density_id_drake.py
- R_EPS_DIAG 뒤에: USE_FORCE_ROWS=True, ROWS_PER_DIR=6, TOTAL_MASS_KG=None,
  noise_diag(sigma_f, sigma_t, use_force), rebuild_noise(sigma_f, sigma_t)
  [R_EPS_DIAG / R_STACK_DIAG / W_HALF 를 반드시 함께 재생성], set_torque_only(enabled).
- regressor()/measure(): USE_FORCE_ROWS False 면 토크 행만.
- constrained_map(): `// 6` -> `// ROWS_PER_DIR`.

## my_work/density_id_objects.py
- set_measurement_averaging 의 alg.R_EPS_DIAG/R_STACK_DIAG/W_HALF 직접 생성 3줄을
  alg.rebuild_noise(_BASE_SIGMA_F*scale, _BASE_SIGMA_T*scale) 한 줄로.

## my_work/design_core.py
- torque_rows_only(value, n_dir) 추가, measurement_equation 의 vector() 에서 사용
  (6축 -> 3축 슬라이싱은 여기 한 곳에서만; 측정·전송·기록은 6축 유지).
- grasp_columns: USE_FORCE_ROWS False 면 np.zeros((3,3)) 블록 제거.
- grasp_map / tls_map 에 grasp_mu_m=None 인자: 사전평균·상자 중심·잔차를 그 둘레로.

## my_work/dual_view.py
- 인자: --total-mass-kg, --torque-only/--use-force(dest torque_only, 기본 None), --grasp-mu-mm X Y Z.
- parse_args 직후(다른 어떤 regressor 호출보다 먼저): torque_only None 이면 hardware=="real",
  alg.set_torque_only(...), real 인데 total_mass_kg 없으면 parser.error.
- prepare(total_mass_kg=None): prior=="weight" 에서 값이 있으면 assembled_mass_kg(spec, rho_gt)
  대신 사용, alg.TOTAL_MASS_KG 에 기록, 출처를 화면에 표시.
- PlannerScreen: 클래스 기본값 grasp_mu_m=None, fixed_mass_kg=None; __init__ 에
  total_mass_kg, grasp_mu_m; fixed_mass_kg 가 있으면 라운드마다 총질량 재유도 금지;
  grasp_map/tls_map 호출에 grasp_mu_m 전달; read_one 의 시뮬 오차 주입 인덱스를
  USE_FORCE_ROWS 에 따라 [3:6]/[0:3].
- 예보 루프의 alg.R_EPS_DIAG 직접 생성 -> alg.noise_diag(...), finally 뒤 alg.rebuild_noise().

## my_work/pivot_ui.py
- "--prior", "water" -> "weight".
- 옵션 루프에 ("--total-mass-kg","TOTAL_MASS_KG"); GRASP_MU_MM 있으면 --grasp-mu-mm;
  USE_FORCE 참이면 --use-force.

## tools/preflight.py
- check_estimator_conf(report, conf) 추가하고 main 에서 check_calibration 다음에 호출:
  GRASP_FRAME!=measured -> FAIL, TOTAL_MASS_KG 없음/비양수 -> FAIL,
  measured 인데 GRASP_MU_MM 있음 -> FAIL, GRASP_SIGMA_MM<15 -> WARN.

## setup/experiment.conf.example
- TOTAL_MASS_KG=0.571, GRASP_MU_MM= (비움), # USE_FORCE=1, GRASP_SIGMA_MM=15.0 로.

## 검증 (반드시 다 돌리고 보여줘)
1) 힘 블록 rank 확인:
   A6 = alg.regressor(np.array([0.4,-0.7])); F = np.vstack([A6[6*i:6*i+3] for i in range(3)])
   sv = np.linalg.svd(F, compute_uv=False); rank 1, 행공간 기저와 VOLUMES 방향 |cos| = 1.0
2) alg.set_torque_only(True) 후 regressor/measure/W_HALF 길이가 3*len(G_DIRS) 로 일치
3) unittest 32개 OK, hardware_real.self_check 13/13, pivot_ui --dry-run OK
4) preflight 새 행 4개 렌더링
5) 복원(5부위 시뮬, 파지점 173.9 mm 어긋뜨림, grasp_map):
     6축/토크만 + 명목 0 -> railing [-11,50,-50], 총질량오차 19~26 %
     6축/토크만 + 명목값 -> 파지점 5 mm 안, 총질량오차 0.03 % / 0.00 %
   남은 ~56 % 는 사전분포 수축(재분배 폭 x10 이면 4~6 %) — 추정기 결함 아님

## 하지 말 것
- 하드웨어·타어·전송 경로에서 렌치를 3축으로 줄이지 마 (원시 렌치 보존, |F| ~ M g 검산).
- SIGMA_F / SIGMA_T 값 자체는 건드리지 마.
- GRASP_FRAME=measured 에서 GRASP_MU_MM 에 173.9 mm 벡터를 넣지 마 (이중 계산).
```

---

## 장비 없이 통합 UI 로 먼저 한 번 (5~15분) — 아무 PC 에서

브랜치를 가져온 뒤, 로봇·카메라·센서 없이 **같은 통합 UI** 로 데스크 램프 실험을
끝까지 돌려볼 수 있습니다. 로봇 PC 가 아니어도 되고, 로봇 PC 에서 해도 로봇에
아무 명령도 보내지 않습니다 (`ROBOT_HOST` 가 비어 있어야만 뜹니다). 렌치·자세·타어는 모의, 파지는 자산의 짐작값입니다.

```bash
./setup/quickstart_sim.sh --check    # 브랜치·Drake 환경·conf·dry-run 만
./setup/quickstart_sim.sh            # 대시보드 http://localhost:8080 — 버튼은 사람이
./setup/quickstart_sim.sh --auto     # 버튼까지 자동, 끝까지 혼자 돈다
```

`setup/experiment.sim.conf` 는 첫 실행 때 예시에서 만들어집니다. `ROBOT_HOST` 가
비어 있어야만 뜹니다 — 값이 있으면 실물로 갈 뻔한 것이라 멈춥니다.

Claude Code 에 시킬 때:

```
./setup/quickstart_sim.sh --auto 를 돌리고, 대시보드 주소와 세션 폴더를 알려줘.
끝나면 sessions/<세션>/experiment_results.json 의 converged, 라운드 수,
부위 질량(link_3/link_1/link_2)을 GT 85/399/87 g 와 나란히 보여줘.
실패하면 어느 단계(0~5)에서 멈췄는지와 터미널 마지막 30줄을 보여줘.
```

---

## 적용 후 실물 실행 — 로봇 PC 에서 한 줄

```bash
./setup/quickstart_real.sh --mass 0.571     # 저울 kg, 힌지 포함
```

브랜치가 없으면 가져오고, 기존 `experiment.conf` 의 키(`TOTAL_MASS_KG`,
`GRASP_FRAME=measured`, `TARE_MODE=gravity`, `GRASP_SIGMA_MM≥15`, `GRASP_MU_MM` 비움)만
고치고, 타어가 없거나 게이트(0.5 N)에 떨어지면 **물어보고** `wrist_tare.sh --run` 을
돌린 뒤, 런처 점검을 거쳐 통합 UI 를 띄웁니다. 탐색 1라운드가 끝나면 UI 가
`tools/round_check.py` 로 검산 3개(힘 크기 / 잔차팽창 / 파지 오프셋)를 찍습니다.
`--check` 를 붙이면 띄우지 않고 준비만 점검합니다.

사람이 남는 일: 저울, 파지(손), 각도 맞추기(손), 경로 승인, 그리고 파지 직후
`tools/check_grasp_frames.py` 회전 차이 확인.

수동으로 할 때:

```bash
./setup/launch_experiment.sh --check
./setup/launch_experiment.sh
```

`dual_view` 를 직접 돌릴 때:

```bash
python dual_view.py --mode deploy --hardware real \
    --object desklamp --grasp pinch --grasp-part link_3 \
    --grasp-frame measured \
    --total-mass-kg <저울값 kg> \
    --grasp-sigma-mm 15 \
    --prior weight
```

**`--grasp-frame measured` 에서는 `--grasp-mu-mm` 을 주지 않습니다** (실측 파지가
이미 기하에 들어가므로 이중 계산). legacy 프레임에서만 씁니다.

`GRASP_SIGMA_MM` 을 10 이 아니라 **15** 로 두는 이유: FoundationPose 위치 오차
10 mm 에 손‑눈 변환 오차(자세 1° → 팔 0.5 m 에서 8.7 mm)가 더해집니다.
표식으로 실측한 값이 있으면 그 값을 쓰십시오.
