# 부위 이름 대응 — 램프

**결론부터. 배달물의 파일 이름이 틀렸습니다.** `link_1_head_...ply` 는 head 가
아니라 **베이스**이고, `link_2_base_...ply` 는 베이스가 아니라 **Head** 입니다.
`link_3_support` 만 이름이 맞습니다.

이걸 모르고 팀원 산출물(FoundationPose 각도, MeshPCA 치수)을 이어 붙이면
**조용히 틀린 밀도가 나옵니다.** 예외도 오류도 안 납니다.

## 대응표

| PIVOT `link_N` | 배달물 파일 이름 | **실제 부위** | 팀원(MeshPCA) 이름 |
| --- | --- | --- | --- |
| `link_1` | `link_1_head_watertight_rgb.ply` | **베이스** (책상에 닿는 판) | `base` |
| `link_2` | `link_2_base_watertight_rgb.ply` | **Head** (갓) | `head` |
| `link_3` | `link_3_support_watertight_rgb.ply` | **연결부/팔** | `support` |

## 무엇으로 확정했나 — 셋이 따로 같은 답을 냅니다

**① 가장 큰 평면.** 베이스는 책상에 놓이므로 큰 평면이 있어야 합니다.
법선이 같은 방향(0.985 이상)인 삼각형 넓이를 모아 잰 값입니다.

```
                                   최대 평면      부피
PIVOT  link_1_head_watertight       199.6 cm^2   441.0 cm^3   <- 베이스
PIVOT  link_2_base_watertight        62.4         165.7
PIVOT  link_3_support_watertight     76.5         219.6

팀원   base_metric_watertight       192.8 cm^2   272.8 cm^3   <- 베이스
팀원   head_metric_watertight        72.3          66.4
팀원   support_metric_watertight     76.6          87.6
```

`link_1_head` 199.6 ↔ 팀원 `base` 192.8 이고, `link_3_support` 76.5 ↔ 팀원
`support` 76.6 은 거의 같은 값입니다.

**② 스캔 좌표계에서의 위치.** 세 부위의 도심 y 가 같은 순서로 늘어섭니다.

```
PIVOT   link_2_base 233.7  <  link_3_support 383.6  <  link_1_head 524.9
팀원    head        198.2  <  support        380.5  <  base        505.0
```

**③ `figures/desklamp_density_render.py`.** 협업 전에 이미 같은 결론에
도달해 있었습니다 — "바로잡은 부위 매핑 ... link_1 = 베이스(바닥),
link_3 = 연결부, link_2 = Head", 위 방향은 베이스 최대 평면(205.9 cm^2)의
법선으로 잡았습니다.

## 관절 대응

PIVOT 은 잡는 부위를 뿌리로 트리를 다시 세웁니다. 기본값
(`--grasp-part link_3`, 연결부를 잡음)에서는 이렇게 됩니다.

```
PIVOT                                        팀원 FoundationPose
joint_2_3 : link_3 -> link_2                 support_head_deg
            (연결부 -> Head)                  (support 와 head 장축 사이 각)
joint_3_1 : link_3 -> link_1                 base_support_deg
            (연결부 -> 베이스)                 (base 와 support 장축 사이 각)
```

**부호와 영점은 아직 안 맞췄습니다.** 팀원 각도는 "부호 없는 기하 각"이라고
`foundationpose/README.md` 에 적혀 있고, PIVOT 관절각은 URDF 축 기준의 부호
있는 값입니다. 실물에서 한 자세를 두 방법으로 읽어 비교해야 확정됩니다.
그 전에는 `adjust_by_pose` 가 목표와 반대로 판정할 수 있습니다.

## 부피가 다릅니다 — 밀도에 직접 옵니다

같은 부위인데 부피가 1.6~2.5 배 다릅니다 (위 표). PIVOT 메시는 정점이
623~1061 개인 거친 워터타이트 껍질이고, 팀원 메시는 36k~47k 개에
MeshPCA 실측 치수로 리사이즈까지 거친 것입니다.

밀도는 `질량 / 부피` 라 **부피가 2 배 다르면 밀도도 2 배 다릅니다.** PIVOT
안에서는 `rho_gt = 실측질량 / 스캔부피` 로 정의해 두어 자기들끼리는
일관되지만(`desk_lamp.build_spec` 주석의 "유효 밀도"), 팀원 메시로 갈아타면
`GROUND_TRUTH` 의 기준 밀도가 전부 바뀝니다.

**어느 쪽이 맞는지는 아직 안 정했습니다.** MeshPCA 가 원본 Depth 로 치수를
재는 것이 목적이므로 팀원 쪽이 metric 으로는 더 믿을 만해 보이지만,
갈아타려면 `desk_lamp.GROUND_TRUTH` 를 같이 다시 잡아야 합니다.

## 팔 단면 — `grippers.py` 표의 70 mm 는 AABB 값입니다

`grippers.py` 머리말 표는 램프 연결부를 "최소 단면 70 mm" 로 적어 두고
그래서 PGC-140(개구 53 mm)으로는 못 잡는다고 결론냅니다. 그런데 그 70 mm 는
**굽은 팔의 AABB** 이고, 같은 파일이 "AABB 는 실제 물어야 할 폭보다 훨씬
크다" 고 스스로 적어 두었습니다.

장축을 따라 얇은 판으로 실제 단면을 재면 이렇습니다.

```
팀원 support (36,894 정점)   장축 322 mm 전 구간에서  9~12 mm x 32~38 mm
PIVOT link_3 (623 정점)      같은 방법으로            16~23 mm (정점이 적어 튐)
```

**연결부는 납작한 바이고 10~23 mm 입니다.** 두 그리퍼 모두 물 수 있습니다.
그리퍼 선택의 근거로 그 표를 쓰면 안 됩니다.

## 팀원 저장소에만 있는 것

`~/MeshPCA/pivot/tare_real.py` 는 `hw.Rb5Driver(ip, enable_motion=True,
max_speed_deg_s=..., max_accel_deg_s2=...)` 를 부르는데, PIVOT 의
`hardware.Rb5Driver` 는 아직 `NotImplementedError` 입니다. **구현이 팀원
PC 에만 있습니다.** 받아 와야 3~6 단계가 이 저장소에서 돕니다.

PIVOT 쪽 실물 경로는 `hardware_real.RbpodoBackend` 라는 다른 설계이고
서명도 다릅니다. 둘 중 하나로 합쳐야 합니다.
