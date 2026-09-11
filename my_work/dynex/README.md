# dynex — 관절체의 동적 여기 궤적 설계와 10-파라미터 관성 식별

논문 PIVoT 의 정적 파이프라인(관절 형상 선택 + 중력 3방향 정지 측정)을 **동적**으로
확장한다. 부위마다 질량 1, 1차 모멘트 3, 관성 텐서 6 의 열 개를 추정하고, 물체의
마찰 관절이 측정 중 미끄러지지 않도록 **힌지 축 토크**를 제약으로 건 채 정보량이
최대인 로봇 궤적을 설계한다. 부족한 관측성은 관절 형상을 바꿔 가며 채운다.

Scalable Real2Sim(Pfaff et al., 2025)의 골격(푸리에 여기 궤적 + 제약 최적화 +
유사관성 양정치 투영)을 따르되, 세 가지가 다르다.

| | Scalable Real2Sim | dynex |
|---|---|---|
| 센서 | iiwa 관절 토크 (로봇 전체를 먼저 식별) | 손목 F/T (센서 아래 툴만 식별) |
| 페이로드 | 강체 하나 | 마찰 관절체 — 형상 θ 마다 다른 강체 |
| 제약 | 관절·충돌 | + **힌지 축 토크 예산**, 파지 마찰 용량, 케이블 |
| 목적 | 조건수 + E-최적 | 사전분포·잡음 모형 기반 정보이득 (논문 식 6 의 동적판) |
| 외부 루프 | 없음 | 관절 형상 선택 (정보이득) → 궤적 재설계 |

## 1. 수식

측정 프레임 S 는 AFT200 퍽 중심. S 에 고정된 강체(툴 + 물체)에 센서가 가하는 렌치:

```
f = m (a_o − g) + α × h + ω × (ω × h)
τ = h × (a_o − g) + I α + ω × (I ω)          [f; τ] = Y(a_o, ω, α, g) · φ,  φ ∈ R^10
```

부위 j 의 φ_j (자기 몸체 프레임) 는 형상 θ 에서 S 로 선형 변환된다:
`φ_S = Σ_j M_j(θ) φ_j`. 그래서 렌치는 Φ = [φ_1; …; φ_P] 에 선형이고, 한 형상에서
관측되는 것은 조립체의 10개뿐이다 → **부위 P 개를 다 풀려면 형상이 최소 P 개**.

힌지 i 가 마찰로 버텨야 하는 토크는 축 성분 하나다:

```
τ_i(t) = â_i · [ τ_o − r_h × f ]_(하류 부위만)   = g_i(t)ᵀ Φ
|g_i(t)ᵀ μ| + k √(g_i(t)ᵀ Σ g_i(t)) ≤ τ_hold,i / 안전계수
```

"관절과 평행하면 빨리, 수직이면 천천히"는 이 식의 결과다. g_i 가 0 인 운동 방향
(관절 축 방향 병진, 관절 축에 수직인 축 둘레 회전 등)은 제약에 안 걸려 최적화가
그쪽 진폭을 키운다. `hardware.hinge_sensitivity` 가 그 방향들을 숫자로 보여 준다.

## 2. 모듈

| 파일 | 내용 | 검증 |
|---|---|---|
| `inertial.py` | φ, 유사관성 J(φ), 프레임 변환 10×10 | Drake `SpatialInertia.Shift/ReExpress` 와 1e-16 일치 |
| `regressor.py` | Y(a,ω,α,g), 힌지 축 토크 행 | Drake `CalcInverseDynamics` 와 1e-16 일치 |
| `kinematics.py` | RB5+AFT200+2F-85+물체 씬에서 S 의 (a_o, ω, α, g), 힌지 기하, 충돌 거리 | 가속도는 유한차분과 일치 |
| `trajectory.py` | 시작·끝 속도·가속도 0 인 주기 푸리에 궤적 | 경계조건·미분 자기검사 |
| `design.py` | 정보이득 최대화, 제약: 관절/속도/가속/작업공간/충돌/힌지/파지/케이블 | 조밀 격자 검증 + 진폭 축소 |
| `identify.py` | 가우시안 사후분포, 영점 주변화, 유사관성 ⪰ 0 투영 (Clarabel) | |
| `simulate.py` | 데이터시트 잡음·분해능·기록 속도·시각 지터·툴 오차 | |
| `configs.py` | 형상 후보 → 대리 점수 → 궤적 설계 → 측정 → 갱신 | |
| `hardware.py` | RB5 / AFT200 / 2F-85 한계, 자유 부분공간, 정보이득 비교 | |

## 3. 실행

```bash
cd my_work
R=../robot_learning/scripts/run_drake_env.sh
$R python -m dynex.run_demo --object 3link --quick                       # 5분
$R python -m dynex.run_demo --object 3link --rounds 4 --variant shell \
    --sensor datasheet_50hz --hardware-report --out results/dynex/run.json
```

주요 옵션: `--hinge-torque`(유지토크, 기본 0.5 N·m) `--safety`(1.5) `--sensor`
(`datasheet_50hz | datasheet_200hz | datasheet_1khz | paper`) `--variant`
(`uniform | shell | insert`: 참값 무게중심을 외형 중심에서 옮긴다) `--densities`
(부위 밀도, 기본은 제작품 실측 347 442 425) `--grip-force`(205 N) `--tool-error`
(툴 파라미터 식별 오차 주입) `--jitter-ms` `--lag-ms`.

자기검사:

```bash
$R python -c "from dynex import inertial, regressor; print(inertial.self_test(), regressor.self_test())"
```

## 4. 실물에 붙일 때 해야 하는 것 (코드가 가정으로 둔 것)

- **툴 동적 식별**: 물체 없이 같은 벌림량에서 여기 궤적 몇 개를 돌려 그리퍼의
  10개를 먼저 푼다 (`identify.update` 를 부위 1개로 쓰면 된다). 툴/물체 질량비가
  2.7 이라 툴 질량 오차 1 % 가 물체 질량 2.7 % 로 샌다.
- **센서 기록**: 지금 드라이버는 Modbus 50 Hz 폴링이고 타임스탬프가 없다. 관절 상태와
  같은 시계로 찍어야 한다. EtherCAT 판은 1 kHz 에 가속도계·자이로가 있어
  (a_o, ω) 를 직접 줄 수 있다.
- **로봇 궤적 실행**: `rbpodo.move_servo_j` 를 5 ms 주기로 보내고, 실제로 실행된
  `jnt_ang` 을 저장해 그것으로 회귀행렬을 만든다 (지령이 아니라 실행값).
- **힌지 유지토크**: 토크 게이지 실측. 정적 자세에서 버틴 최대 토크가 하한이다.
- **미끄러짐 검사**: 궤적 전후의 관절각을 카메라로 비교. 바뀌었으면 그 궤적을 버리고
  예산을 낮춘다.

## 5. 시뮬레이션 결과 (2026-09-11, 3링크 제작품 밀도 347/442/425, 셸 도심 어긋남, 4라운드)

`results/dynex/v3_*.json|log|png`. 라운드 = 형상 선택 → 정적 3방향(자세당 2 s) → 여기 궤적(8 s).

| 조건 | 정적 IG(1라운드) | 동적 IG(1라운드) | 최종 질량 오차 % | 최종 σ_mass % | 최종 σ_I % |
|---|---|---|---|---|---|
| AFT200 50 Hz | 21.1 | 3.2 | 4.4 / 6.8 / 5.2 | 16 / 16 / 14 | 56 / 61 / 76 |
| AFT200 1 kHz | 34.8 | 3.3 | 7.5 / 12.1 / 9.4 | 17 / 15 / 15 | 61 / 63 / 74 |
| 논문 잡음 모형 | 52.5 | 3.4 | 1.4 / 10.7 / 13.7 | 12 / 13 / 14 | 54 / 46 / 50 |
| 형상 고정(강체 baseline) | 21.1 | 3.2 | 6.7 / 9.5 / 6.5 | 18 / 18 / 16 | 52 / 60 / 70 |
| 힌지 제약 없음 | 21.1 | 3.2 | 4.9 / 7.5 / 5.6 | 16 / 16 / 14 | 57 / 64 / 77 |

손으로 만든 극단 궤적(손목 291 °/s, 한계·충돌·힌지 전부 위반)도 σ_mass 12 %, σ_I 45~70 %
에서 멈춘다. 원인은 `identifiability.py` 가 보여 주는 구조적 널공간(30개 중 8개)이다.
GIVENS_AND_RISKS.md 0절 참고.

하드웨어 수치 (같은 궤적의 정보이득): 50 Hz 23.5 nat, 200 Hz 28.7, 1 kHz 35.6, 논문 잡음
56.4. 관성 텐서 항만의 토크 최대 0.0005 N·m 로 센서 분해능 0.015 N·m 의 1/30.

## 6. 통합 UI 로 돌리기 (2026-09-11 추가)

`setup/experiment.conf` 에 `METHOD=dynamic` 을 넣으면 `pivot_ui.py` 가 4단계(탐색)에서
`dual_view.py` 대신 `dynex.session` 을 라운드마다 부른다. 세션 폴더 규약은 그대로다.

| 파일 | 내용 |
|---|---|
| `angle_round_N.json` | 추천 θ, 측정 θ (deg). 다음 라운드 추천은 이전 라운드가 남긴다 |
| `path_round_N.json` | 정적 3자세 관절각, 여기 궤적 경유점(100 Hz), 예측 최대값(힌지 비율·속도·간격·파지 토크) |
| `wrench_round_N.csv` | t, q(6), 원시 렌치(6), 툴을 뺀 렌치(6) — 오프라인 재추정용 |
| `posterior_round_N.json` | 부위별 추정·σ, 식별 가능 좌표의 σ(정적/관성), 수렴 여부 |
| `dynex_state.npz` | 사후분포 (라운드 사이 이어 쓰기) |
| `export/asset_dynamic.json` | 최종 파라미터 + 안 보이는 조합 8개 (사전분포로 정해진 방향) |

```bash
# 장비 없이 끝까지 (모의 장비: AFT200 사양 잡음 100 Hz, 각도 오차 5 %, 툴 오차)
$R python pivot_ui.py --conf ../setup/experiment_sim.conf --sim --auto
# 세션만 따로
$R python -m dynex.session --object 3link --hardware sim --rounds 3 --auto
# 실물 (미검증): 툴 파일·저울 총질량·추적기 파일이 필요
$R python -m dynex.session --hardware real --tool-file calibration/tool_dynamic.json \
    --total-mass-kg 0.343 --pose-file /tmp/.../latest.json --pose-keys joint1 joint2
```

실물 훅은 `hardware_real.py` 끝에 있다 (`rb5_stream`, `Aft200Stream`, `align_and_regress`).
**장비 없이 작성했으므로 미검증**이다. 첫 시험은 물체 없이, 작은 진폭부터.

## 7. 힌지는 이렇게 다룬다

- 질량: 관절마다 41 g (저울 실측, `MEASURED_HINGE_KG`). dynex 는 이것을 **자식 링크의
  핀 축 위 점질량**으로 참값·사전분포 양쪽에 넣는다. 축 위 질량은 부모/자식 어느 쪽에
  두어도 측정과 시뮬레이션이 같으므로(안 보이는 조합), 이 선택은 부위별 표의 약속일 뿐이다.
- 부피 7.45 cm³ 는 논문 표에서 온 값이고 저장소 안에 재현 경로가 없다(`MEASURED_HINGE_CM3`
  주석). 밀도 환산에만 쓰인다.
- 유지토크: **몰라도 된다 (자동 모드).** 세션은 "이미 버틴 최대 토크"를 인증값으로 기억하고,
  다음 자세·궤적은 그 1.5배(`--hinge-growth`)까지만 시도한다. 처음에는 0.05 N·m
  (`--hinge-floor-nm`)부터. 실행 뒤 카메라 각도가 2° 이상 움직였으면 미끄러진 것으로 보고
  그 데이터를 버리고 상한을 그 토크의 0.8배로 내린다. 축과 평행한 운동은 예산을 거의 안 쓰므로
  자연히 먼저 열리고, 버틴 것이 확인될수록 다른 방향이 열린다. 토크 게이지 값이 있으면
  `HINGE_TORQUE` 로 주고, 그러면 그 값/1.5 를 넘지 않는 상한으로만 쓰인다.
  실물 실험 코드(dual_view, 정적)는 기본값에서 힌지 검사를 안 한다.
- 무게중심 오프셋(`hinge_com_offset_mm`) 기본 0: 날개가 한쪽으로 뻗은 실제 힌지와 다르지만,
  축에서 벗어난 몫은 측정으로 보정된다.
