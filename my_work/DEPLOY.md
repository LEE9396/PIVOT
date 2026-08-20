# 실물 배포 — 팀원 PC 에서 준비하기

> **센서가 오기 전에 여기까지는 다 해 둘 수 있습니다.**
> 장비 없이 실물 경로 전체를 리허설하는 모드가 있습니다.

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

## 2. 장비 없이 지금 할 수 있는 것 세 가지

### (a) 안전 로직 자가검증 — 10초

```bash
$R python hardware_real.py --check
```

속도 상한, 준정적 이동 시간, 도착 검증, 서보 상태기계, 렌치 이상치 제거,
센서 부호 규약, 자세 stale 검사, 타어링 유효기간 — **열 가지를 장비 없이
시험합니다.** 10/10 이 나와야 합니다.

### (b) 실물 경로 리허설 — 가짜 팔로 전 과정

```bash
$R python dual_view.py --mode deploy --arm-backend fake --object 3link
```

타어링 → 작업자 UI → 안전 확인 → 이동 → 측정 → 갱신이 **실제와 같은
순서로** 돕니다. 장비만 가짜입니다. 절차를 몸에 익히는 데 쓰세요.

### (c) 이론 검증 — 알고리즘이 정말 맞는지

```bash
$R python study_theory.py        # 정리 1~3 을 기계로 확인
$R python export_urdf.py --object 3link   # 파이프라인 끝단까지
```

## 3. 센서가 오면 채울 것 — **여섯 곳뿐입니다**

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
   a) set_servo / halt 만 연결  ->  --arm-backend rbpodo 로 배선 확인
   b) joint_positions 연결      ->  팔을 손으로 옮기며 값이 맞는지 눈으로 확인
   c) move_to 는 1~2도 이동부터  ->  비상정지를 손에 들고
```

### (2) F/T — `dual_view._ft_sample_fn`

AFT200 스트림에서 한 샘플 `[Fx Fy Fz Tx Ty Tz]` (N, N·m) 를 돌려주는 함수.
평균·이상치 제거·부호 검사는 `Aft200Sensor` 가 이미 합니다.

### (3) 자세 — `dual_view._pose_fn`

FoundationPose 에서 `(관절각 deg, sigma deg)` 를 돌려주는 함수.
차원·유한성·stale 검사는 `FoundationPoseSensor` 가 이미 합니다.

## 4. 실물 실험 전 반드시 하는 것

| | 무엇 | 왜 |
| --- | --- | --- |
| 1 | **핸드아이 캘리브레이션** (`calibrate_camera.py`) | 안 하면 도면상 명목 위치로 자세를 계산합니다 |
| 2 | **타어링** (물체를 잡기 **전에**) | 그리퍼가 물체보다 무겁습니다. 중력 방향별로 따로 |
| 3 | **센서 부호 확인** (`check_sign_convention`) | 축이 뒤집히면 밀도가 통째로 틀어지는데 결과만 봐선 모릅니다 |
| 4 | **저울로 총질량** | 사전분포의 출발점. 힌지까지 올린 값 |

## 5. 실물 측정 순서 — **커스텀 물체 먼저**

```
   1순위  custom 2-link   미지수 3, 1 라운드. 가장 단순 -> 파이프라인 디버깅
   2순위  custom 3-link   미지수 5, 2 라운드. 이론 검증의 핵심
   3순위  desk lamp       분해할 때 배수법으로 부위 부피도 실측
   4순위  laptop          기지 추 50 g 차분 (부위 귀속 검증)
```

램프·노트북을 먼저 하면 오차가 **재구성 탓인지 코드 탓인지 못 가립니다.**
CAD 정답이 있는 물체에서 코드를 먼저 검증하는 것이 순서입니다.

## 6. 두 PC 로 나눠 돌릴 때

한 대로 되면 나눌 필요 없습니다 (`--bus local`). 나눌 때는 **양쪽 플래그가
같아야** 합니다 — 다르면 계획 쪽이 검사한 장면과 로봇 쪽이 경로를 계획하는
장면이 조용히 달라집니다.

```bash
# 작업 PC
$R python dual_view.py --mode deploy --bus tcp --object 3link --gripper robotiq2f85

# 로봇 PC
$R python robot_node.py --host <작업PC IP> --object 3link --gripper robotiq2f85
```

## 7. 막히면 볼 곳

| 증상 | 어디 |
| --- | --- |
| pydrake 임포트 실패 | 래퍼(`run_drake_env.sh`) 없이 실행했는지 |
| 도착 실패 예외 | `move_to` 가 블로킹이 아닌 것. `hardware_real.Rb5Driver.follow` 주석 |
| 힘 방향 경고 | 센서 프레임 축 순서. `check_sign_convention` |
| 타어링 만료 | 30분 지남. 다시 재세요 (온도 드리프트) |
| 알고리즘이 이해 안 됨 | [ALGORITHM.md](ALGORITHM.md), [THEORY.md](THEORY.md) |
