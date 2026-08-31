# AI 도우미에게 (Claude Code / Codex / ChatGPT 등)

이 파일은 **사람이 아니라 AI 가 읽는 안내서**입니다.
새 PC 에서 이 저장소를 받은 사람이 AI 에게 "환경 구성해 줘" 라고 하면,
AI 는 이 파일만 읽고 끝까지 갈 수 있어야 합니다.

사람이 할 일은 이 두 줄이 전부입니다.

```bash
git clone https://github.com/LEE9396/PIVOT.git ~/Desktop/PIVOT && cd ~/Desktop/PIVOT
# 그 다음 AI 에게: "AGENTS.md 를 읽고 이 PC 에 환경을 구성해 줘"
```

---

## 이 저장소가 무엇인가

로봇이 관절 달린 물체를 **잡고 자세를 바꿔 가며 손목 힘을 재서**,
부위별 밀도(=질량)를 알아내고 시뮬레이션용 URDF 를 만드는 연구 코드입니다.
자세한 것은 [README.md](README.md), 검증 방법은
[my_work/VERIFICATION.md](my_work/VERIFICATION.md) 에 있습니다.

---

## 환경 구성 — 이대로 하세요

```bash
cd <저장소 루트>
./setup/bootstrap.sh
```

이 한 줄이 파이썬 3.12 환경 생성 + 꾸러미 설치 + 자가 진단까지 합니다.
5~10 분 걸립니다. **직접 pip 를 부르거나 conda 환경을 새로 만들지 마세요.**

진단만 다시 하려면:

```bash
./setup/bootstrap.sh --check
```

### 반드시 지켜야 하는 규칙 넷

1. **파이썬을 직접 부르지 않는다.** 모든 실행은 이 앞머리를 붙입니다.

   ```bash
   cd my_work
   ../robot_learning/scripts/run_drake_env.sh python <스크립트>
   ```

   이 래퍼가 ROS 2 가 전역에 깔린 PC 에서 `PYTHONPATH`/`LD_LIBRARY_PATH` 오염을
   걷어냅니다. 그냥 `python` 을 부르면 pydrake 임포트가 조용히 깨집니다.

2. **작업 디렉터리는 `my_work/`.** 스크립트들이 서로를 `import` 로 부르므로
   다른 곳에서 부르면 모듈을 못 찾습니다.

3. **Drake 는 1.54.0 으로 고정.** 올리지 마세요. IK·거리제약 API 가 바뀝니다.

4. **가상환경(`.venv-*`)은 저장소에 넣지 않는다.** 753 MB 입니다. 각 PC 에서
   bootstrap 이 만듭니다.

---

## 잘 됐는지 확인하는 법

```bash
cd my_work
../robot_learning/scripts/run_drake_env.sh python ../setup/doctor.py
```

11 개 항목이 모두 `[통과]` 면 끝입니다. 실패하면 무엇을 고치면 되는지
진단이 직접 알려 줍니다. `-v` 를 붙이면 파이썬 역추적까지 나옵니다.

포트 7000/7001 항목은 **다른 화면이 이미 떠 있으면 실패로 나옵니다.**
그건 고장이 아닙니다.

---

## 대표 실행 명령

| 하려는 것 | 명령 (모두 `my_work/` 에서) |
| --- | --- |
| 시뮬레이션 검증 (램프) | `../robot_learning/scripts/run_drake_env.sh python dual_view.py --mode sim --object desklamp` |
| 시뮬레이션 검증 (3링크) | `... python dual_view.py --mode sim --object 3link` |
| 실물, 작업 PC 쪽 | `... python dual_view.py --mode deploy --object 3link --bus tcp --bus-port 5555` |
| 실물, 로봇 PC 쪽 | `... python robot_node.py --host <작업PC IP> --port 5555 --hardware real` |
| 배선만 확인 | 위 로봇 PC 명령에서 `--hardware sim` |
| 설계 선택지 실험 | `... python study_tls.py` (study_*.py 여섯 개) |

화면은 브라우저에서 `http://localhost:7000`, `http://localhost:7001` 로 엽니다.
원격 PC 라면 `ssh -L 7000:localhost:7000 -L 7001:localhost:7001 사용자@PC`.

---

## 코드를 고칠 때 알아야 할 것

### 구조

```
my_work/
  design_core.py        추정·설계의 핵심 (후보 고르기, TLS, 정지 판단)
  density_id_objects.py 물체 정의와 Drake plant
  density_id_drake.py   회귀행렬·측정 모형
  robot_scene.py        RB5+AFT200+그리퍼+테이블 씬, IK, 충돌 검사
  dual_view.py          전체를 묶는 실행 파일 (화면 2개, sim/deploy 모드)
  robot_node.py         실물 로봇 쪽 프로세스 (TCP 반대편)
  pose_bus.py           두 프로세스 사이 JSON 한 줄 규약
  hardware.py           실물 장비 드라이버 자리 (지금은 비어 있음)
  gripper_hw.py         Robotiq 2F-85 실물 드라이버 (USB/Modbus-RTU)
  density_view.py       탐색 결과를 메시 색으로 비교하는 화면
  desk_lamp.py          스캔한 램프를 물체로 물리는 코드
  mesh_props.py         메시에서 부피·도심·관성·색을 뽑는 코드
  grippers.py           PGC-140 / Robotiq 2F-85 정의
  study_*.py            설계 선택지 비교 실험 여섯 개
```

### 이 저장소의 약속

- **GT(정답)는 채점에만 씁니다.** 탐색·정지 판단에 GT 를 쓰면 연구가 무의미해집니다.
  실물에서는 GT 를 모르기 때문입니다. `desk_lamp.GROUND_TRUTH` 가 쓰이는 곳은
  `validate()` 의 오차 계산뿐이어야 합니다.
- **정지 조건은 불확실성 기반.** 목표는 사후 반폭이며, 치우침 몫을 더해 봅니다.
- **주석은 한국어**, 이유를 적습니다. "무엇을 하는지"는 코드가 이미 말합니다.
- 새 물체를 넣을 때는 `density_id_objects.OBJECTS` 에 `ObjectSpec` 을 더합니다.

### 실물 장비를 붙일 때

`hardware.py` 의 `Rb5Driver`, `Aft200Sensor`, `FoundationPoseSensor`, `run_tare`
네 곳이 `NotImplementedError` 와 함께 **무엇을 채우면 되는지** 적혀 있습니다.
그 클래스들만 채우면 나머지 코드는 그대로 돕니다.

---

## 하지 말아야 할 것

- `pip install` 을 저장소 밖 전역 파이썬에 하기 (ROS 2 환경을 망가뜨립니다)
- Drake 버전 올리기
- `third_party/` 안을 고치기 (원본 자산입니다. 고칠 일이 있으면 `my_work/` 에서 감싸세요)
- `assets/desk_lamp_minimal_sim/` 안을 고치기 (협업자가 준 스캔 배달물입니다)
- `third_party/` 안의 자산을 다른 곳에 퍼 나르기 — 원저작자 것이고 일부는
  재배포 라이선스가 확인되지 않았습니다 (`third_party/HTD/THIRD_PARTY_ASSETS.md`).
