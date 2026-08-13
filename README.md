# 잡아 보고 무게를 알아내기 — 관절 물체의 부위별 밀도 식별

로봇이 관절 달린 물체를 **잡고 자세를 바꿔 가며 손목 힘/토크를 재서**,
부위마다의 밀도(따라서 질량)를 알아내고 시뮬레이션에 바로 쓸 수 있는
URDF 를 내놓는 연구 코드입니다.

저울로 통째 무게를 재면 **총합**밖에 모릅니다. 이 파이프라인은 물체를
여러 자세로 세워 손목이 받는 힘을 여러 번 읽고, 그 차이에서 **부위별로**
나눕니다. 자세를 아무렇게나 고르지 않고, 다음에 어느 자세를 재야 가장 많이
알게 되는지를 계산해서 고릅니다.

```
스캔한 물체 모형 ──> 다음 자세를 고른다 ──> 잡을 수 있나 검사 ──> 손목 힘 측정
                            ^                                        |
                            └──────── 아직 흐릿하면 한 번 더 ─────────┘
                                              |
                                    충분히 또렷하면 URDF 로 내보냄
```

**처음이라면 [my_work/VERIFICATION.md](my_work/VERIFICATION.md) 를 먼저 보세요.**
시뮬레이션 검증과 실물 로봇 검증, 두 길을 그림과 함께 단계별로 설명합니다.

---

## 빨리 시작하기

```bash
git clone <이 저장소 주소> ~/Desktop/HTD-main
cd ~/Desktop/HTD-main
./setup/bootstrap.sh                    # 파이썬 3.12 + Drake 1.54 + 자가 진단

cd my_work
../robot_learning/scripts/run_drake_env.sh python dual_view.py \
    --mode sim --object desklamp        # 브라우저로 localhost:7000, 7001
```

자세한 설치·문제 해결은 [SETUP.md](SETUP.md).
AI 에게 시킬 거라면 [AGENTS.md](AGENTS.md) 를 읽히세요.

---

## 지금까지 나온 결과

실물 스캔한 스탠드 램프(3부위, 총 562 g)를 시뮬레이션에서 돌린 결과입니다.
정답은 램프를 분해해 저울로 잰 값이며 **채점에만** 씁니다.

| 부위 | 실측 질량 | 추정 오차 | |
| --- | ---: | ---: | --- |
| 베이스 | 396.0 g | 0.01 % | |
| 연결부(Arm) | 82.0 g | 0.21 % | 로봇이 잡는 곳 |
| Head | 84.0 g | 0.03 % | |

2 라운드 만에 목표 불확실성(상대반폭 1 %)에 도달했습니다.
후보 자세 25 개 중 20 개가 도달·충돌·경로 검사를 모두 통과했습니다.

설계 선택지들은 각각 실험으로 비교했습니다 (`my_work/study_*.py`).

| 물음 | 답 | 근거 |
| --- | --- | --- |
| 후보를 격자에서 고를까, 연속 최적화할까 | **연속 최적화** | `study_continuous.py` |
| 어느 방향으로 기울일 때 정보가 가장 느나 | 직교 3방향이면 회전은 무관 | `study_tilt.py` |
| 최소제곱 vs 총최소제곱 | **TLS** (각도 오차의 치우침을 없앰, 1.41 % → 0.01 %) | `study_tls.py` |
| D / A / E 중 무엇 | **D-최적** (부위 수를 늘려도 안정) | `study_criterion.py` |
| 언제 멈추나 | 불확실성 기반 + 치우침 몫 | `study_stopping.py` |
| 무거운 힌지(41 g)는 무시해도 되나 | 안 됨 — 부위로 세면 됨 | `study_hinge.py` |

---

## 무엇이 들어 있나

```
my_work/          알고리즘·화면·실물 연결 (연구 코드는 전부 여기)
  VERIFICATION.md   두 검증 방법 설명   <- 먼저 읽기
  ALGORITHM.md      알고리즘 설명 (고등학생 수준)
  README.md         코드 파일 하나하나의 설명
setup/            새 PC 환경 구성 (bootstrap.sh, doctor.py)
assets/           협업자가 준 스탠드 램프 스캔 배달물
robot_learning/   Drake 환경 실행기와 RB5 씬 유틸
third_party/      RB5·PGC·AFT200 (HTD) / Robotiq 2F-85 원본 자산
docs/, progress/  이전 검증 기록
```

## 장비

| | |
| --- | --- |
| 팔 | Rainbow Robotics RB5-850E |
| 손목 F/T | AIDIN AFT200-D80-C |
| 그리퍼 | Robotiq 2F-85 (기본) / DH PGC-140-50 |
| 카메라 | Intel RealSense D456 (관절각 관측: FoundationPose) |
| 시뮬레이터 | Drake 1.54 |

---

## 비공개로 두세요

`third_party/HTD/THIRD_PARTY_ASSETS.md` 가 PGC-140-50 과 AFT200 자산에 대해
"연구 수령자 밖으로 넘기기 전에 재배포 허락을 확인하라"고 적어 두었습니다.
이 저장소는 **비공개(private)** 로 유지하고, 팀원 초대로만 공유하세요.
