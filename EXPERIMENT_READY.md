# 실물 실험 준비 완료 보고서

대상 물체 세 개 — **2-link**, **3-link**, **desk lamp** — 로 실물 실험을 할 수
있도록 코드·UI·검증을 마쳤다. 이 문서 하나로 무엇을 어떻게 하는지 알 수 있다.

작성 2026-09-02 · 검증은 전부 시뮬레이션(Drake)에서 실제로 돌려 확인했다.

---

## 0. 무엇이 막혀 있었나

2026-09-02 실물 실험이 시작조차 못 했다. 원인이 **두 곳에 독립적으로** 있었고
둘 다 **조용했다** — 예외도 경고도 없었다.

```
창 2   SAM3 마스크가 base 를 0 픽셀로 잡음 -> FoundationPose 시작 못 함
       -> 런처가 180 초 기다리다 종료
창 1   head 가 AFT200 마운트를 41.2 mm 관통
       -> "힌지·도달·충돌을 모두 통과하는 자세가 없다"
```

세 중력방향에서 관통 깊이가 **소수점까지 같았다**(-41.232475 mm). 로봇 자세 q 와
무관하다는 뜻이고, 그리퍼-물체 **용접 변환**에 박힌 값이라는 증거다. 경로 계획
문제가 아니었다.

---

## 1. 실험 준비 코드 — 무엇을 만들었고 어떻게 쓰나

### 1.1 새로 만든 도구 (`tools/`)

| 도구 | 무엇 | 로봇 이동 |
| --- | --- | --- |
| `preflight.py` | 0단계 준비 점검. 조용히 틀리던 것을 OK/주의/실패로 | 없음 |
| `verify_objects.py` | 세 물체가 실제로 도달·충돌·경로를 통과하는지 | 없음 |
| `make_convex.py` | 통짜 충돌 메시를 볼록 조각으로 (CoACD) | 없음 |
| `make_part_legend.py` | 부위 이름표 그림 (창 2 마스크용) | 없음 |
| `angle_signs.py` | FoundationPose 각도의 부호·영점 + **실제 각도 오차** | 없음 |
| `grasp_measure.py` | 파지 변환 `X_G_O` 측정 (짐작 대체) | 없음 |
| `pivot_session.py` | 세션 폴더 + 단계(phase) 상태 | 없음 |
| `import_easyhec.py` | EasyHeC `Tc_c2b` → PIVOT `X_WC` | 없음 |

### 1.2 고친 것

| 파일 | 무엇 | 왜 |
| --- | --- | --- |
| `desk_lamp.py` | 볼록 분해 없으면 **예외** (조용한 폴백 제거) | 사고가 조용했던 이유 |
| | `LAYOUT=="final"` pinch 보정 복원 | 파지점이 팔 한가운데로 갔다 |
| `robot_scene.py` | `GRASP_LONG_AXIS_BY_OBJECT = {"desklamp":"y"}` | 갈래 구조는 z 로 두면 못 푼다 |
| | 책상 **실측 평면** 로더 (없으면 크게 알림) | 도면값으로 조용히 돌았다 |
| | **측정 파지 변환** 훅 (`load_measured_grasp`) | 짐작을 측정으로 |
| | 각도 여유 게이트 `PIVOT_ANGLE_MARGIN_DEG` (기본 0) | 창 2 각도 오차 1~3° 반영 |
| `dual_view.py` | 검사를 **실행 사슬**과 일치시킴 | 별 모양으로 검사하고 사슬로 실행했다 |
| | 직접 간선이 막히면 home 경유 | 라운드가 통째로 날아갔다 |
| | 실물 도착 자세에서 간격 확인 | 실물엔 이동 중 감시가 없었다 |
| | 미검증 각도 추천 차단 (`NoFeasibleAngle`) | 사람이 미검증 각도를 맞추고 있었다 |
| `launch_experiment.sh` | 부위 이름표 자동 생성, 수동 마스크 기본 | 창 2 가 먼저 죽었다 |
| `make_desk_lamp_masks.py` | 마스크 창에 **부위 이름표** 나란히 | 어느 덩어리가 base 인지 몰랐다 |
| `tare_real.py` | `--manual` (직접교시), `--verify` (드리프트 확인) | 세팅 때 1번으로 |

### 1.3 준비 절차 — 순서대로

**A. 장비 없이 (오늘)**

```bash
# 1) 물체 충돌 형상을 볼록 조각으로
python -m pip install coacd
$R python tools/make_convex.py <자산>/collision_meshes --report

# 2) 세 물체가 도달·충돌·경로를 통과하는지
$R python tools/verify_objects.py --json results/verify.json

# 3) 준비 점검
$R python tools/preflight.py --conf setup/experiment.conf
```

`$R` 은 `../robot_learning/scripts/run_drake_env.sh` 다. **맨 python 으로 부르면
pydrake 가 조용히 깨진다.**

**B. 실물 앞, 로봇 정지**

```bash
# 4) 핸드아이 — DGIST EasyHeC (Robotiq 자산판) 로 캡처·최적화 후
$R python tools/import_easyhec.py --input <EasyHeC>/models/.../Tc_c2b.txt \
    --intrinsics <EasyHeC>/data/.../K.txt \
    --output calibration/camera_cam_d456_front.json

# 5) 책상 평면 (카메라만, 로봇 명령 0개)
$R python calibrate_table_rgbd.py --handeye calibration/<핸드아이>.json \
    --output calibration/rb5_table_current.json

# 6) 창 2 를 띄워 마스크를 뚫고, 각도 부호·영점
./setup/launch_experiment.sh
$R python tools/angle_signs.py --pose-file /tmp/lamp_foundationpose_live/latest.json \
    --joint 1 --seconds 40 --output calibration/angle_signs.json
```

`angle_signs` 가 내는 **잔차를 `ANGLE_FLOOR_DEG` 에 넣는다.** 그게 창 2 의 실제
각도 오차다.

**C. 로봇 이동 — 세팅 때 한 번**

```bash
# 7) 3자세 타어 (직접교시. 로봇에 명령하지 않는다)
PIVOT_WORKDIR=$PWD $R python ~/MeshPCA/pivot/tare_real.py --manual --setup \
    --output calibration/aft_tare_current.json --overwrite
```

**D. 실험 직전 — 10초**

```bash
# 8) 타어 드리프트 확인. 문턱 안이면 세팅 때 값을 그대로 쓴다
PIVOT_WORKDIR=$PWD $R python ~/MeshPCA/pivot/tare_real.py --verify \
    --output calibration/aft_tare_current.json
```

---

## 2. 통합 UI 설계

`my_work/pivot_ui.py` — 창 1 이 지휘자이고 단계 0~5 를 하나로 묶는다.

### 2.1 핵심 두 가지

**로봇은 나중에 들어온다.** 0~2 단계는 물체만 올린다. 예전에는 `prepare()` 가
시작하자마자 로봇 씬을 세워서, 파지점 짐작이 어긋나면 **시작조차 못 했다.**

**용접 변환은 짐작이 아니라 측정이다.** 1단계 「파지 완료」에서 창 2 의 물체 자세
+ 핸드아이 + 로봇 q 로 `X_G_O` 를 계산해 세션에 남기고, 3단계에서 그 값으로
용접한다.

```
X_G_O = FK(q)⁻¹ · X_W_C · X_C_O
         ↑          ↑        ↑
      로봇 관절각  핸드아이  FoundationPose
```

`robot_scene.build_scene` 이 `PIVOT_GRASP_FILE` / `PIVOT_SESSION/grasp.json` 을
읽어 그 값을 `WeldFrames` 에 쓴다. 그러면 `GRASP_LONG_AXIS`, `grasp_rotation`,
볼록 조각 정점평균 같은 **짐작이 전부 안 쓰인다.**

### 2.2 단계

| 단계 | 씬 | 하는 일 | 넘어가는 조건 |
| --- | --- | --- | --- |
| 0 준비 | 물체만 | `preflight.py` 결과 표시 | 실패 0 |
| 1 파지점 | 물체만 | 파지 후보 표시 → 사람이 창 3 으로 물기 | 「파지 완료」 → `X_G_O` 측정 |
| 2 각도 | 물체만 | 정보이득 최대 θ 추천, 창 2 각도와 비교 | 「각도 조정 완료」 |
| 3 경로 | **로봇 씬** | 측정 파지로 용접, 도달·충돌·RRT | 「승인」 — 로봇이 움직이는 유일한 문 |
| 4 탐색 | 로봇 씬 | 중력 3방향 측정, 창 3·4 갱신 | 목표 안이면 5, 아니면 2 |
| 5 내보내기 | — | 밀도 반영한 URDF | 세션 폴더에 자산 |

### 2.3 창 2·3 과의 연결

창 1 이 세션 폴더의 `phase.json` 을 쓰고, 창 2·3 은 그것을 폴링해 표시를 바꾼다.
파일 하나라 파이썬 환경이 달라도(창 2 는 conda, 창 3 은 venv) 그냥 된다.

```
session_YYYYMMDD_HHMM/
  phase.json          지금 단계
  preflight.json      0단계 점검
  grasp.json          X_G_O, 파지 부위, 개구량
  angle_round_N.json  추천 θ, 확정 θ, 제외 목록
  path_round_N.json   경로와 충돌 검사
  wrench_round_N.csv  렌치 원시값
  posterior_round_N.json
  export/             sim-ready 자산
```

### 2.4 되돌아가는 길

| 어디서 | 왜 | 어디로 |
| --- | --- | --- |
| 3 경로 | 충돌 또는 도달 불가 | 2 각도 또는 1 재파지 |
| 4 탐색 | 불확실성이 목표 밖 | 2 각도 |

---

## 3. 검증 결과 — 시뮬레이션에서 실제로 돌렸다

각도 격자 9점(2link 는 3점), 중력 3방향, 최소 여유 10 mm, RRT 사슬 경로.

### 3.1 세 물체

| 물체 | 사슬 | 파지 단면 | 장축 | 도달·충돌 | 최소 간격 | 경로 |
| --- | --- | --- | --- | --- | --- | --- |
| 2-link | parent → child | 40.0 mm | z | **2/3** | 11.00 mm | 사슬 연결 |
| 3-link | link0 → link1 → link2 | 44.0 mm | z | **8/9** | 11.00 mm | 사슬 연결 |
| desk lamp | link_3 → link_1 → link_2 | 23.6 mm | **y** | **6/9** | 11.00 mm | 사슬 연결 |

막힌 각도는 전부 **완전히 접힌 자세**(2link θ=180°, 3link θ₁=180°, 램프
θ₂=−111.4°)다. 물체 자기 링크끼리 닿는 자세라 물리적으로 당연하다.

경로는 세 물체 모두 `start → g1 → g2 → g3 → start` 사슬이 이어졌다.

### 3.2 램프 — 고친 것이 실제로 효과인가

같은 격자에서 조건만 바꿔 재봤다.

| 조건 | 도달 | 최소 간격 |
| --- | --- | --- |
| ① 전부 적용 (조각 + pinch 보정 + `long_axis=y`) | **6/9** | 11.0 mm |
| ② `long_axis` 를 z 로 | **0/9** | — |
| ③ 볼록 분해 없이 통짜 메시 | **6/9** | 11.0 mm |
| ④ 둘 다 없음 (2026-09-02 실패 상태) | **0/9** | — |

**효과를 낸 것은 `long_axis="y"` 하나다.** ③ 이 ① 과 같은 것은, pinch 보정을
복원하면서 파지점이 조각 수와 무관해졌기 때문이다(체적 중심을 직접 쓴다).

볼록 분해는 이 자세들에서는 안 걸리지만 **여전히 넣어야 한다** — 충돌 형상이
1.6~2.6 배 부풀고, 다른 자세에서는 걸린다.

### 3.3 도구 검증

| 도구 | 결과 |
| --- | --- |
| `grasp_measure --self-test` | 위치 오차 **2.6e-13 mm**, 자세 오차 1.7e-06° — 변환 순서 정확 |
| `angle_signs` 합성 검증 | 잡음 2.0° 주입 → 잔차 **2.25°** 회수 (잔차 = 실제 각도 오차) |
| `angle_signs --model-only` | 램프 joint_1: directed 규약이 **단조**(꺾임 0) → 한 점으로 q 확정 가능 |
| `preflight` | 8항목 판정. 자산·볼록분해·빌드일치 통과, 실물 캘리브레이션만 실패(정상) |
| `pivot_session` | 단계 전이·원자적 쓰기 통과 |
| `pivot_ui --dry-run` | 단계 기계 0~5 통과 |
| `make_part_legend` | 램프 3부위 렌더 — 평판=base, 굽은팔=support, 라이트바=head |
| `make_convex` | 실제 메시 분해. 조각이 굵으면 경고 |
| `import_easyhec` | 팀원 문서의 실제 행렬로 검산 — 위치·거리·하향각 일치 |
| `robot_scene` 책상 로더 | 명목 750 mm 왕복 일치, 기울기 반영, 없으면 경고 |
| `robot_scene` 파지 훅 | 5경로(인자/env 2종/배열/경로) 전부 통과 |

### 3.4 아직 검증 못 한 것

- `pivot_ui` 1·3 단계의 **화면** — Meshcat 뷰와 로봇 씬 구성은 실물에서 처음 뜬다
- `export_urdf.py` 인자 규약 — 5단계에서 실패하면 여기
- `tare_real --manual/--verify` 의 **장비 통신 부분** — `rbpodo.CobotData` 응답
- 팀원 PC 의 `tare_real.py` 는 essential.zip 의 것과 **다른 세 번째 버전**이다
  (`joint_deg`, `repeat_delta` 를 쓴다). 패치를 거기에 다시 맞춰야 한다

---

## 4. 남아 있는 판단 사항

| | 내용 |
| --- | --- |
| 램프 자산 | 이 검증은 **08-25 빌드**(`Lamp_Final.zip`)로 했다. 실물은 **09-01 빌드**를 쓴다. 관절 원점이 50~90 mm 다르므로 그쪽에서 다시 `verify_objects.py` 를 돌려야 한다 |
| `ANGLE_TOL_RAD` | IK 가 1.000° 경계에 붙어 푼다. 타어와 실험에서 각각 1° → 최대 2° 불일치 = 렌치의 4 %. 조일 값어치 있음 |
| 타어 90° 의문 | 기록 관절각으로 복원한 방향이 라벨과 90° 어긋난다. 새로 `--setup` 하면 `direction_error_deg` 로 판별된다 |
| 타어 모형 잔차 | `raw = 무게·ĝ + 영점` 이 4.35 N 안 맞는다(램프 무게의 78 %). 케이블 당김 의심. 자세별 표를 쓰는 한 상쇄되므로 지금은 무해 |

---

## 5. 팀원 PC 환경 구성 프롬프트

별도 저장소에 환경을 세울 때 그대로 붙여 넣는다.

````text
PIVOT 실물 실험 환경을 이 PC 에 새로 구성해 줘. 세 물체(2-link, 3-link,
desk lamp)로 실험할 거야.

━━ 저장소 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  git clone -b real-experiment-ready https://github.com/LEE9396/PIVOT.git ~/PIVOT
  git clone https://github.com/Yuseong-Cheon/MeshPCA.git ~/MeshPCA

  절차는 ~/PIVOT/EXPERIMENT_READY.md 에 전부 있다. 그걸 읽고 따라가 줘.
  저장소 규칙은 AGENTS.md, 부위 이름 대응은 my_work/NAMING.md 에 있다.

━━ 규칙 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ★ PIVOT 의 모든 python 실행에는
      ../robot_learning/scripts/run_drake_env.sh
    를 앞에 붙인다. 맨 python 이면 pydrake 가 조용히 깨진다.
  ★ Drake 는 1.54.0 고정. 올리지 마라.
  ★ 로봇을 실제로 움직이는 명령은 반드시 나한테 확인받고 실행해라.

━━ 순서 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 환경
   ./setup/bootstrap.sh
   $R python setup/doctor.py                # 전부 통과해야 함

2. 물체 자산
   - 2-link, 3-link 는 코드에 상수로 들어 있다. 할 일 없다.
   - 램프는 배달물이 필요하다. 자산 폴더를 이렇게 만들어라:
       <lamp>/standlamp*.urdf
       <lamp>/visual_meshes/{base,support,head}.obj
       <lamp>/collision_meshes/{base,support,head}.obj
       <lamp>/collision_meshes/convex/<부위>/part_NN.obj   <- 없으면 만들어라
     조각이 없으면:
       python -m pip install coacd
       $R python tools/make_convex.py <lamp>/collision_meshes --report
     "가장 긴 조각이 부위의 절반을 넘는다" 경고가 나오면 --threshold 를
     절반으로 낮춰 다시 구워라.

3. 설정
   cp setup/experiment.conf.example setup/experiment.conf
   ★ LAMP_ASSET_DIR 과 FP_MESH_DIR 은 **반드시 같은 트리**를 가리켜야 한다.
     하나만 바꾸면 충돌 기하와 창 2 오버레이가 50~90 mm 다른 좌표계가 된다.

4. 장비 없이 검증 — 여기까지는 로봇 없이 다 된다
   $R python tools/verify_objects.py --json results/verify.json
     통과 기준: 세 물체 모두 도달 각도가 1개 이상, 사슬 경로 연결
   $R python tools/preflight.py --conf setup/experiment.conf
     실패가 0 이 될 때까지 고쳐라 (캘리브레이션 항목은 5번 뒤에 통과한다)
   $R python my_work/pivot_ui.py --dry-run
   $R python my_work/hardware_real.py --check

5. 캘리브레이션 — 순서가 중요하다
   a) 핸드아이  DGIST EasyHeC (Robotiq 자산판) 로 캡처 10~20 장 → 최적화
                $R python tools/import_easyhec.py --input .../Tc_c2b.txt \
                    --intrinsics .../K.txt --output calibration/camera_<id>.json
   b) 책상      $R python my_work/calibrate_table_rgbd.py --handeye ... \
                    --output calibration/rb5_table_current.json
                (a 가 있어야 돈다. 로봇에 명령하지 않는다)
   c) 마스크    ./setup/launch_experiment.sh 로 창 2 를 띄운다.
                부위 이름표가 자동으로 구워지고 수동 박스 모드로 뜬다.
                latest.json 이 나오면 통과.
   d) 각도부호  $R python tools/angle_signs.py --pose-file <latest.json> \
                    --joint 1 --seconds 40 --output calibration/angle_signs.json
                관절마다 한 번씩. 나온 **잔차를 ANGLE_FLOOR_DEG 에 넣어라.**
   e) 타어      ★ 여기서 로봇이 처음 움직인다. 나한테 확인받아라.
                PIVOT_WORKDIR=$PWD $R python ~/MeshPCA/pivot/tare_real.py \
                    --manual --setup --output calibration/aft_tare_current.json
                직접교시로 자세를 만들면 로봇에 명령하지 않는다.

6. 실험
   $R python my_work/pivot_ui.py --conf setup/experiment.conf
   또는 기존 4창:  ./setup/launch_experiment.sh
   실험 직전마다:  tare_real.py --verify   (10초, 드리프트 확인)

━━ 막히면 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  "자세가 없다"        -> tools/verify_objects.py 로 어느 쌍이 몇 mm 겹치는지 봐라
  창 2 가 안 뜬다      -> /tmp/pivot_win2.log. 마스크 실패면 MANUAL_MASK=1
  결과가 조용히 이상   -> preflight 의 **주의** 항목을 먼저 봐라.
                         명목값으로 돌고 있다는 뜻이다.

  ★ preflight 에서 주의가 뜨면 "일단 돌려보자"가 아니라
    "결과를 믿을 수 없다"로 읽어라.
````

---

## 6. 사용자 안내서

`docs/pivot_experiment_guide.html` — 로봇도 프로그래밍도 처음인 사람이 읽고
실험을 준비할 수 있게 쓴 문서. 준비 6단계, 실험 5단계, 조용히 틀리는 것들,
용어 사전.
