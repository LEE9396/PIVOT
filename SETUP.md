# 환경 구성 — 작업 PC / 로봇 PC

이 저장소를 받은 두 대의 PC에서 각각 무엇을 설치하고 어떻게 확인하는지 적었습니다.
**두 PC 모두 같은 설치를 합니다.** 다르게 하는 것은 마지막에 실행하는 명령 하나뿐입니다.

| | 작업 PC | 로봇 PC |
| --- | --- | --- |
| 하는 일 | 다음 자세를 **고르고**, 충돌을 **검사**하고, 밀도를 **추정** | 실제 RB5를 **움직이고**, AFT200 힘을 **읽음** |
| 필요한 것 | 이 저장소 + Drake 환경 | 이 저장소 + Drake 환경 + 로봇 드라이버 |
| 로봇 연결 | 필요 없음 | 필요함 (RB5 / AFT200 / 카메라) |
| 화면 | [1] 계획·탐색 | [2] 작업자 UI |
| 실행 명령 | `dual_view.py --mode deploy --bus tcp` | `robot_node.py --host <작업PC IP>` |

> 시뮬레이션만 해 볼 거라면 **PC 한 대**로 충분합니다. 아래 3단계까지만 하고
> [VERIFICATION.md](my_work/VERIFICATION.md)의 시뮬레이션 명령을 쓰세요.

---

## 0. 준비물

| | 최소 | 확인 방법 |
| --- | --- | --- |
| OS | Ubuntu 22.04 / 24.04 | `lsb_release -a` |
| 디스크 | 3 GB (저장소 0.1 GB + 파이썬 환경 0.8 GB) | `df -h .` |
| 파이썬 | **3.12** 를 만들 수 있으면 됨 | 아래 bootstrap이 알아서 함 |
| 인터넷 | 처음 설치할 때만 | |

파이썬 3.12는 conda나 `python3.12-venv` 중 **아무거나** 있으면 됩니다.
둘 다 없으면 bootstrap이 어느 쪽을 설치하면 되는지 알려줍니다.

---

## 1. 받기

```bash
git clone https://github.com/LEE9396/PIVOT.git ~/Desktop/PIVOT
cd ~/Desktop/PIVOT
```

폴더 이름은 아무거나 좋습니다. 코드가 자기 위치를 스스로 찾습니다.

---

## 2. 설치 (한 줄)

```bash
./setup/bootstrap.sh
```

이것이 하는 일은 넷입니다.

1. `robot_learning/.venv-drake-1.54-py312/` 에 파이썬 3.12 환경을 만든다
2. `robot_learning/requirements/drake.txt` 를 그대로 설치한다 (Drake 1.54, numpy, scipy 등)
3. 저장소 안 자산이 다 있는지 본다 (RB5 URDF, Robotiq 메시, 램프 스캔)
4. 자가 진단을 돌린다

5~10분 걸립니다 (대부분 Drake 내려받기). 마지막에 이렇게 나오면 성공입니다.

```
모두 통과했습니다. 다음으로 무엇을 할지는 SETUP.md 4장을 보세요.
```

---

## 3. 잘 됐는지 확인

```bash
./setup/bootstrap.sh --check      # 설치는 건너뛰고 진단만
```

또는 직접:

```bash
cd my_work
../robot_learning/scripts/run_drake_env.sh python ../setup/doctor.py
```

진단이 보는 것:

| 항목 | 왜 보나 |
| --- | --- |
| Drake 1.54 임포트 | 버전이 다르면 IK API가 달라집니다 |
| scipy | 총최소제곱(TLS) 추정기가 씁니다 |
| RB5 / PGC / AFT200 자산 | 해시까지 맞춰 봅니다 |
| Robotiq 2F-85 URDF | 기본 그리퍼입니다 |
| 램프 스캔 배달물 | 시뮬레이션 검증 대상입니다 |
| 3-link 물체 씬 | Drake가 실제로 씬을 세우는지 |
| 포트 7000/7001 | 화면 두 개가 쓸 자리 |

---

## 4. 무엇을 실행하나

### 시뮬레이션 검증 (PC 한 대, 데스크 램프)

```bash
cd ~/Desktop/PIVOT/my_work
../robot_learning/scripts/run_drake_env.sh python dual_view.py \
    --mode sim --object desklamp
```

**터미널에 뜨는 두 주소를 브라우저 탭 두 개로 여세요.**

```
[1] 계획·탐색 화면      http://localhost:7000     ← 알고리즘이 자세를 고르는 곳
[2] 로봇 + 작업자 UI    http://localhost:7001     ← RB5가 램프를 들고 도는 곳
```

둘 다 연 뒤 [2] 화면의 **`두 화면을 모두 연 뒤 눌러 시작`** 버튼을 누르면
시작합니다. 준비(후보 자세 계산)에 40~60초 걸리고, 그 뒤 한 라운드가
1~2분입니다. 보통 **2라운드**에서 끝납니다.

한 라운드에서 이런 순서로 움직입니다.

```
파지 자세 → 관절각 조정 → 각도 측정 자세 → 중력 3방향 측정 → 복귀
```

램프는 두 관절 축이 거의 나란해서 각도 측정 자세가 **하나로 묶입니다.**
게다가 파지 자세가 이미 잘 보는 편이라(4.4 px/deg) 그냥 그 자리에서 읽고
넘어가는 라운드도 있습니다 — 그럴 때는 `파지 자세가 이미 충분해 그대로
읽습니다` 라고 찍힙니다. 3-link는 축이 90° 어긋나 자세가 둘로 나뉩니다.

끝나면 부위별 밀도와 **실측 GT 대비 오차**가 찍힙니다. 마지막에 URDF를
만들지 물어봅니다.

```
  link_3 (연결부)  밀도  374.2 +/- 2.3 kg/m^3   [GT  373  오차 0.21%]
  link_2 (베이스)  밀도 2389.7 +/- 0.8 kg/m^3   [GT 2390  오차 0.01%]
  link_1 (Head)    밀도  190.4 +/- 0.3 kg/m^3   [GT  190  오차 0.03%]
```

**눈으로 볼 때 쓸 만한 옵션**

```bash
# 사람이 슬라이더로 각도를 맞추는 대신 자동으로 (구경만 할 때)
... --auto-adjust

# 로봇 움직임을 느리게 (기본 8초, 12초면 여유 있게 보입니다)
... --move-duration 12

# 브라우저를 열 새도 없이 바로 시작 (녹화·로그용)
... --autostart --auto-adjust

# 램프 색이 보랏빛으로 보이는 게 거슬리면 (스캔에 조명색이 구워져 있습니다)
DESK_LAMP_WHITE_BALANCE=1 ... python dual_view.py --mode sim --object desklamp
```

**숫자만 빠르게 보고 싶다면** 화면 없이 도는 것도 있습니다.

```bash
../robot_learning/scripts/run_drake_env.sh python desk_lamp.py          # 8번 반복 채점
../robot_learning/scripts/run_drake_env.sh python lamp_view.py          # 램프만 3D로
```

### 실물 로봇 검증 (PC 두 대, 커스텀 물체)

**작업 PC 터미널** — 먼저 켭니다. 로봇 쪽이 붙을 때까지 기다립니다.

```bash
cd my_work
../robot_learning/scripts/run_drake_env.sh python dual_view.py \
    --mode deploy --object 3link --bus tcp --bus-port 5555
```

**로봇 PC 터미널** — 작업 PC의 IP를 적어 줍니다.

```bash
cd my_work
../robot_learning/scripts/run_drake_env.sh python robot_node.py \
    --host 192.168.0.10 --port 5555 --hardware real
```

IP는 작업 PC에서 `hostname -I` 로 확인합니다. 방화벽이 막으면:

```bash
sudo ufw allow 5555/tcp        # 작업 PC 에서
```

**장비 없이 배선만 확인**하려면 로봇 PC에서 `--hardware sim` 을 씁니다.
로봇을 움직이지 않고 가짜 힘을 돌려주므로, TCP가 통하는지만 봅니다.

```bash
# 같은 PC에서 두 터미널로도 됩니다
../robot_learning/scripts/run_drake_env.sh python robot_node.py \
    --host 127.0.0.1 --port 5555 --hardware sim
```

---

## 5. 로봇 PC에만 필요한 것

`--hardware real` 은 아직 **비어 있습니다**. `my_work/hardware.py` 의 네 곳을
그 PC의 드라이버로 채워야 실물이 움직입니다.

| 채울 것 | 무엇을 해야 하나 |
| --- | --- |
| `Rb5Driver` | RB5-850E 에 관절각을 보내고 도착을 기다림 |
| `Aft200Sensor` | AFT200 에서 6축 렌치를 읽음 |
| `FoundationPoseSensor` | 카메라로 물체 관절각을 읽음 |
| `run_tare` | 물체 없이 그리퍼만 든 상태의 렌치를 미리 재둠 |

각 클래스는 지금 `NotImplementedError` 와 함께 **무엇을 채우면 되는지**를
말해 줍니다. 그대로 따라가면 됩니다. ROS1 노드가 이미 있다면
`pose_bus.Ros1Bus` 골격도 준비돼 있습니다.

타어링(tare)은 건너뛰면 안 됩니다. Robotiq 2F-85 만 400 g이 넘는데
램프 전체가 562 g입니다. 자세한 이유는
[VERIFICATION.md 2장](my_work/VERIFICATION.md)에 있습니다.

---

## 6. 잘 안 될 때

| 증상 | 원인과 해결 |
| --- | --- |
| `Drake environment not found` | 2장의 bootstrap을 안 돌렸습니다 |
| `ModuleNotFoundError: pydrake` | `python` 을 직접 부르고 있습니다. 반드시 `run_drake_env.sh` 를 앞에 붙이세요 |
| `램프 배달물을 못 찾았다` | `assets/desk_lamp_minimal_sim/` 이 있는지 보세요. 다른 곳에 뒀다면 `export DESK_LAMP_DELIVERY=/그/경로` |
| `HTD RB5 URDF not found` | `third_party/HTD/` 가 비었습니다. clone이 덜 됐습니다 |
| 브라우저에 아무것도 안 보임 | 포트 7000/7001을 다른 프로그램이 쓰고 있습니다. `lsof -i:7000` |
| 원격 PC라 localhost가 안 열림 | SSH 포트포워딩: `ssh -L 7000:localhost:7000 -L 7001:localhost:7001 사용자@그PC` |
| TCP 연결이 안 됨 | 작업 PC를 **먼저** 켰는지, IP와 포트가 맞는지, 방화벽이 열렸는지 |
| 램프 색이 보랏빛이다 | 스캔에 촬영 조명색이 구워져 있습니다. `export DESK_LAMP_WHITE_BALANCE=1` 로 보정할 수 있습니다 |

---

## 7. 무엇을 읽으면 되나

| 문서 | 내용 |
| --- | --- |
| [README.md](README.md) | 이 연구가 무엇인지 |
| [my_work/VERIFICATION.md](my_work/VERIFICATION.md) | **두 검증 방법의 단계별 설명** — 먼저 읽으세요 |
| [my_work/ALGORITHM.md](my_work/ALGORITHM.md) | 알고리즘을 그림으로 (고등학생 수준) |
| [my_work/README.md](my_work/README.md) | 코드 파일 하나하나의 설명 |
| [AGENTS.md](AGENTS.md) | AI(Claude/GPT)에게 시킬 때 읽히는 파일 |
