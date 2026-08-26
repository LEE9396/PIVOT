# 실물 배포 — 팀원 PC 에서 준비하고 로봇으로 돌리기

> **센서가 오기 전에 §1~§3 은 다 해 둘 수 있습니다.**
> 장비 없이 캘리브레이션과 탐색 전 과정을 리허설하는 모드가 있습니다.

읽는 순서는 위에서 아래입니다. 캘리브레이션(§3)을 건너뛰면 §6 이 조용히
틀린 값을 냅니다.

---

## 0. 먼저 — 알고리즘에서 바뀐 것 넷

옛 판본으로 돌린 결과가 있으면 **버리고 다시 재야 합니다.** 넷 다 부위가
2·3개일 때는 안 드러나다가 늘면서 드러난 것들입니다.

| 무엇 | 옛 판본의 증상 | 지금 |
| --- | --- | --- |
| 정지 판정이 힌지 행까지 봄 | 부위가 다 수렴해도 **루프가 안 끝남** | 판정에서 힌지 제외 (`stopping_width`) |
| 잔차가 각도 보정 δ 를 미반영 | 팽창 79배. 실제 0.28 % 인데 **89.17 % 로 보고** | `residual_scale` 로 θ+δ 에서 재계산 |
| TLS 반복 상한 400 고정 | 수렴 안 한 답을 수렴한 것처럼 반환 | 상한 해제 (문제 크기에 비례) |
| 목표 반폭이 임의값 | — | **부위 수마다 다르게** 잡는다 (아래) |

**목표 반폭을 부위 수에 따라 정합니다.** 재구성이 이미 질량에 2~3 % 를
기여하므로 그보다 낮은 목표는 갖고 있지 않은 정밀도를 인증하려는 것입니다.

```
   부위 2·3개   1 %      (실물 커스텀 물체)
   부위 4개     1.5 %
   부위 5개     2 %
   부위 6개     2 % 이상, 그리고 각도를 2 % 안으로 읽어야 함
```

> 근거는 `PAPER_HANDOFF.md` 4-10 과 `figures/fig_precision_halfwidth.png` 입니다.
> **각도 정밀도가 유일한 지렛대**라는 것이 이 프로젝트의 결론이고, 그래서
> §5 에 트래커 오차 실측이 새로 들어갔습니다.

---

## 1. 환경

```bash
git clone https://github.com/LEE9396/PIVOT.git ~/Desktop/PIVOT && cd ~/Desktop/PIVOT
./setup/bootstrap.sh              # 최초 1회 (5~10분)
./setup/bootstrap.sh --check      # 12항목 자가 진단
```

모든 실행은 `my_work/` 에서 래퍼로 합니다. 맨 `python` 을 부르면 pydrake 가
조용히 깨집니다.

```bash
cd my_work
R=../robot_learning/scripts/run_drake_env.sh
```

**Drake 는 1.54.0 고정입니다. 올리지 마세요.**

**opencv 는 Drake 환경에 넣지 않습니다.** 카메라로 태그를 '보는' 일에만
필요하고, 그건 시스템 python 에서 도는 별도 프로세스가 합니다 (§3).
FoundationPose 도 같은 구조입니다.

```bash
python3 -m pip install --user opencv-contrib-python   # 시스템 python 쪽에만
```

---

## 2. 장비 없이 지금 할 수 있는 것 셋

### (a) 안전 로직 자가검증 — 10초

```bash
$R python hardware_real.py --check
```

속도 상한, 준정적 이동 시간, 도착 검증, 서보 상태기계, 렌치 이상치 제거,
센서 부호 규약, 자세 stale 검사, 타어링 유효기간 — **열 가지를 장비 없이
시험합니다.** 10/10 이 나와야 합니다.

### (b) 탐색 전 과정 리허설 — 가짜 팔로

```bash
$R python dual_view.py --mode deploy --arm-backend fake --object 3link
```

타어링 → 작업자 UI → 안전 확인 → 이동 → 측정 → 갱신이 **실제와 같은
순서로** 돕니다. 장비만 가짜입니다. 절차를 몸에 익히는 데 쓰세요.

### (c) 이론 검증 — 알고리즘이 정말 맞는지

```bash
$R python study_theory.py                  # 정리 1~3 을 기계로 확인
$R python export_urdf.py --object 3link    # 파이프라인 끝단까지
```

---

## 3. 캘리브레이션 — **건너뛰면 안 됩니다**

### 왜 하나

각도를 읽는 자세는 **카메라가 어디서 보느냐**로 정해집니다. 도면상 명목
위치로 계산해 두고 실제 카메라가 20 cm 옆에 있으면, 애써 고른 자세가 실제로는
최적이 아니고 심하면 물체가 화면 밖으로 나갑니다. 카메라는 팔이 부딪힐 수 있는
**장애물**이기도 해서 충돌 판정도 같이 어긋납니다.

그리고 §0 에서 봤듯 **각도 정밀도가 이 방법의 유일한 지렛대**입니다.
캘리브레이션이 어긋나면 그 지렛대를 통째로 잃습니다.

### 구조 — 프로세스가 둘인 이유

```
   [Drake 환경]  calibrate_camera.py --run
                 팔을 자세로 보내고, FK 로 X_WE 를 만들고, AX=XB 를 푼다
                      |
                      |  {"cmd":"detect"}  →  {"ok":.., "X_CT":..}
                      ↓
   [시스템 python]  calib_detect.py --serve
                 카메라에서 한 장 찍어 ChArUco 자세를 낸다  (opencv 필요)
```

풀이는 numpy 로 닫아 뒀습니다 (`handeye.py`). eye-to-hand 배치입니다 —
카메라는 실험실에 고정돼 로봇을 바라보고, **보정판은 그리퍼에 볼트로 붙어
로봇과 함께 움직입니다.**

### 3-1. 보정판 만들기

```bash
python3 calib_detect.py --make-board board.png \
    --squares-x 7 --squares-y 5 --square-mm 35 --marker-mm 26 --dpi 300
```

**A4 에 100 % 배율로 출력해 평평한 판에 붙입니다.** 프린터가 축소하면 치수가
틀어지고, 그 오차가 그대로 캘리브레이션 오차가 됩니다. 출력 후 자로 한 칸을
재서 35 mm 인지 확인하세요.

판을 그리퍼에 **단단히** 고정합니다. 흔들리면 자세마다 `X_ET` 가 달라져
AX=XB 가 안 풀립니다.

### 3-2. 장비 없이 먼저 리허설

```bash
# 정답을 알고 전 과정을 돈다. 복원 오차가 나오면 배관이 맞는 것
$R python calibrate_camera.py --run --simulate --poses 20

# 자세만 고르고 로봇은 안 움직인다
$R python calibrate_camera.py --run --dry-run --poses 20
```

> `--simulate` 와 `--dry-run` 은 **`--run` 과 같이** 써야 합니다. 혼자 주면
> 지금 설정된 카메라만 출력하고 끝납니다.

카메라도 로봇도 없이 돕니다. **여기서 안 되면 실물에서도 안 됩니다.**
이렇게 나오면 정상입니다.

```
   20/20  코너 24개  재투영 0.300 px

푼 결과 (20자세, 자세쌍 190개)
  자기 일관성  0.78 mm rms / 0.083 deg rms   (최악 1.32 mm)
  보정판 부착   tcp 에서 [3.245  8.109  123.758] mm
  (--simulate 이라 저장하지 않습니다)
  진짜 위치와의 차이 0.43 mm
```

**볼 것은 맨 아랫줄입니다.** 정답을 아는 상태로 돌렸으므로 복원 오차가
1 mm 안쪽이면 배관이 맞는 것입니다.

### 3-3. 실제로 재기 — 터미널 두 개

```bash
# 터미널 1 — 시스템 python. run_drake_env.sh 를 쓰지 않습니다
python3 calib_detect.py --serve --port 5566 \
    --squares-x 7 --squares-y 5 --square-mm 35 --marker-mm 26

# 터미널 2 — Drake 환경
$R python calibrate_camera.py --run --detector-port 5566 --poses 20 \
    --gripper robotiq2f85 --move-duration 8.0
```

먼저 검출이 되는지 눈으로 확인하세요.

```bash
python3 calib_detect.py --preview      # 보드가 잡히면 코너가 그려집니다
```

**자세 20개면 충분합니다.** 회전이 다양해야 AX=XB 가 잘 풀리므로, 스크립트가
자세를 고를 때 회전축이 겹치지 않게 뽑습니다. `--board-margin-mm` 으로 판이
화면 안에 남는 여유를 조절합니다.

### 3-4. 결과 확인

결과는 여기 저장되고, 파일이 있으면 파이프라인이 **자동으로** 씁니다
(`robot_scene.load_camera`).

```
calibration/camera_cam_d456_front.json
```

```bash
# 이 카메라로 각도 측정 자세를 다시 계산해 본다
$R python calibrate_camera.py --check-poses

# 장면에 카메라를 얹어 눈으로 본다
$R python calibrate_camera.py --show
```

**RMS 재투영 오차를 기록해 두세요.** 1 px 를 넘으면 다시 하는 편이 낫습니다.

### 3-5. 손으로 넣기 (다른 도구로 이미 푼 경우)

ROS `easy_handeye`, MoveIt hand-eye 등의 결과도 같은 형태입니다.

```bash
$R python calibrate_camera.py --matrix 0.42 -0.90 ... --rms-px 0.31
$R python calibrate_camera.py --position 0.8 -0.3 0.6 --look-at 0.4 0.0 0.2
```

> Drake 카메라 규약은 **z 가 전방, y 가 아래** — OpenCV 와 같습니다.

---

## 4. 센서가 오면 채울 것 — **여섯 곳뿐입니다**

안전·평균·이상치·단위·타어링은 전부 구현돼 있습니다.
벤더 API 가 들어오는 자리만 채우면 됩니다.

### (1) 팔 — `hardware_real.RbpodoBackend` 의 네 함수

```python
joint_positions()        # -> (6,) rad   ★ 관절 순서가 URDF 와 같은지 확인
move_to(q, duration_s)   # 도착까지 블로킹   ★ 여기가 안전의 핵심
halt()                   # 즉시 정지
set_servo(on)            # 서보 on/off
```

> **rbpodo 는 deg API 가 있습니다.** 이 클래스는 바깥으로 **항상 rad** 를
> 내보내야 합니다. `deg_api=True` 로 두면 `_to_rad/_from_rad` 가 처리합니다.

**처음 붙이는 날 순서** — 이 순서를 지키세요.

```
   a) set_servo / halt 만 연결   →  --arm-backend rbpodo 로 배선 확인
   b) joint_positions 연결       →  팔을 손으로 옮기며 값이 맞는지 눈으로 확인
   c) move_to 는 1~2도 이동부터  →  비상정지를 손에 들고
```

### (2) F/T — `dual_view._ft_sample_fn`

AFT200 스트림에서 한 샘플 `[Fx Fy Fz Tx Ty Tz]` (N, N·m) 를 돌려주는 함수.
평균·이상치 제거·부호 검사는 `Aft200Sensor` 가 이미 합니다.

### (3) 자세 — `dual_view._pose_fn`

FoundationPose 에서 `(관절각 deg, sigma deg)` 를 돌려주는 함수.
차원·유한성·stale 검사는 `FoundationPoseSensor` 가 이미 합니다.

> **sigma 를 성의 있게 채우세요.** 이 값이 `rel_error` 로 들어가 탐색과 정지
> 판정을 좌우합니다. 못 믿을 값을 넣으면 §0 의 지렛대가 헛돕니다.

---

## 5. 실물 실험 전 반드시 하는 것

| | 무엇 | 왜 |
| --- | --- | --- |
| 1 | **핸드아이 캘리브레이션** (§3) | 안 하면 도면상 명목 위치로 자세를 계산합니다 |
| 2 | **트래커 오차 실측** ← 새로 | 고정 장면을 30회 반복 측정해 각도 sigma 를 잽니다. 시뮬 재현(§6-9)에 이 값을 씁니다 |
| 3 | **타어링** (물체를 잡기 **전에**) | 그리퍼가 물체보다 무겁습니다. 중력 방향별로 따로 |
| 4 | **센서 부호 확인** (`check_sign_convention`) | 축이 뒤집히면 밀도가 통째로 틀어지는데 결과만 봐선 모릅니다 |
| 5 | **저울로 총질량** | 사전분포의 출발점. 힌지까지 올린 값 |

**2번이 새로 들어간 이유** — 시뮬레이션은 지금 각도 오차 5 % 를 가정합니다.
실측값으로 바꿔 돌려야 *"기대치"* 가 아니라 *"이 조건에서 나와야 할 값"* 이
되고, 그것과 실물을 비교해야 격차의 의미가 분명해집니다.

---

## 6. 실물 세션 절차 — 물체 하나에 한 번

```
   1.  저울로 총질량                              사전분포
   2.  트래커 오차 실측 (고정 장면 30회)          → sigma_theta [deg]
   3.  타어링 (물체 잡기 전, 중력 방향별로)
   4.  밀도 추정 폐루프                           주 결과
   5.  같은 물체 3~5회 반복                       반복성
   6.  라운드별 시각 로그                         소요 시간 (자동 기록됨)
   7.  관절을 놓고 처지는 영상                    시뮬레이터 재생 검증용
   8.  (램프) 분해 → 저울 + 배수법                GT
       (노트북) 기지 추 50 g 부착 후 4 를 한 번 더  부위 귀속 검증
   ────────────────────────────────────────────────────────────
   9.  세션 후: 같은 모형·같은 각도오차로 시뮬 재실행
```

**9번은 세션 뒤 컴퓨터에서만 하면 됩니다. 추가 장비 시간이 안 듭니다.**
논문 Tab. II 의 `Ours(sim)` 열이 여기서 나오고, 그것이 우리 방법의 **상한**
입니다. 실물이 거기서 벌어진 만큼이 시뮬이 모형화하지 않은 것 전부입니다.

> **성립 조건 둘** — ① 시뮬과 실물이 **같은 스캔 모형**을 써야 합니다.
> 커스텀 물체에 CAD 를 쓰면 격차에 "CAD vs 스캔" 이 섞입니다.
> ② 시뮬의 각도 오차를 2번의 **실측값**으로 맞춰야 합니다.

### 실제로 돌리는 명령

```bash
# 커스텀 3-link, 목표 1 %, 트래커 실측 sigma 가 1.8도였다면
$R python dual_view.py --mode deploy --arm-backend rbpodo \
    --robot-host 192.168.0.10 --object 3link --gripper robotiq2f85 \
    --target 0.01 --angle-floor-deg 1.8 --max-rounds 8 \
    --urdf-out out/3link_measured.urdf
```

```
--target            목표 반폭. §0 의 표대로 부위 수에 맞춰 정한다
--angle-floor-deg   트래커 실측 sigma. 안 주면 기본 0.5도를 쓴다
--max-rounds        예산. 사람이 관절을 맞추므로 라운드당 1~2분
--urdf-out          끝나면 sim-ready asset 을 여기로 뱉는다
--autostart         작업자 확인 없이 바로 시작 (리허설에서만 쓸 것)
```

세션 후 시뮬 재현 (9번):

```bash
$R python dual_view.py --mode sim --object 3link \
    --target 0.01 --angle-error 0.018 --max-rounds 8
```

---

## 7. 측정 순서 — **커스텀 물체 먼저**

```
   1순위  custom 2-link   미지수 3, 1 라운드. 가장 단순 → 파이프라인 디버깅
   2순위  custom 3-link   미지수 5, 2 라운드. 이론 검증의 핵심
   3순위  desk lamp       분해할 때 배수법으로 부위 부피도 실측
   4순위  laptop          기지 추 50 g 차분 (부위 귀속 검증)
```

램프·노트북을 먼저 하면 오차가 **재구성 탓인지 코드 탓인지 못 가립니다.**
CAD 정답이 있는 물체에서 코드를 먼저 검증하는 것이 순서입니다.

---

## 8. 두 PC 로 나눠 돌릴 때

한 대로 되면 나눌 필요 없습니다 (`--bus local`). 나눌 때는 **양쪽 플래그가
같아야** 합니다 — 다르면 계획 쪽이 검사한 장면과 로봇 쪽이 경로를 계획하는
장면이 조용히 달라집니다.

```bash
# 작업 PC
$R python dual_view.py --mode deploy --bus tcp --object 3link --gripper robotiq2f85

# 로봇 PC
$R python robot_node.py --host <작업PC IP> --object 3link --gripper robotiq2f85
```

---

## 9. 막히면 볼 곳

| 증상 | 어디 |
| --- | --- |
| pydrake 임포트 실패 | 래퍼(`run_drake_env.sh`) 없이 실행했는지 |
| `cv2` 없다 | 시스템 python 에 설치. **Drake 환경에 넣지 말 것** (§1) |
| 보정판이 안 잡힘 | `calib_detect.py --preview`. 조명·초점·판 평탄도 |
| AX=XB 가 안 풀림 | 자세 회전이 다양한지. 판이 그리퍼에 단단히 붙었는지 |
| 자세가 화면 밖 | 캘리브레이션을 안 했거나 `--check-poses` 로 재확인 |
| 도착 실패 예외 | `move_to` 가 블로킹이 아닌 것. `hardware_real.Rb5Driver.follow` 주석 |
| 힘 방향 경고 | 센서 프레임 축 순서. `check_sign_convention` |
| 타어링 만료 | 30분 지남. 다시 재세요 (온도 드리프트) |
| **루프가 안 끝남** | 옛 판본인지 확인 (§0). `git log --oneline -1` |
| **반폭이 터무니없이 큼** | 〃. 잔차 팽창 버그는 79배까지 갔습니다 |
| 알고리즘이 이해 안 됨 | [ALGORITHM.md](ALGORITHM.md), [THEORY.md](THEORY.md) |
| 왜 이 설계인지 | [PAPER_HANDOFF.md](PAPER_HANDOFF.md) — 2-b 에 이론/실험 경계 |
