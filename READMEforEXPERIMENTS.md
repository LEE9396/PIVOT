# 실험용 파일 안내 — 무엇을 어떤 순서로 보나

이 브랜치(`real-experiment-ready`)에 실물 실험에 필요한 것이 전부 들어 있다.
파일이 많으니 **어떤 순서로 무엇을 보면 되는지**를 여기 적는다.

> ⚠️ 저장소 기본 브랜치는 `master` 다. 아래 파일들은 이 브랜치에만 있다.
> ```bash
> git fetch origin && git checkout real-experiment-ready
> ```

---

## 가장 짧은 경로

시간이 없으면 이 셋만 하면 된다.

1. **`TEAMMATE_CHECKLIST.md` 를 처음부터 끝까지 읽는다**
2. **★ 표시 9개가 무엇인지 확인한다** — 빠뜨리면 *에러 없이* 틀린 결과가 나오는 것들
3. **`tools/verify_objects.py` 를 돌린다** — 여기서 통과하면 절반은 온 것

```bash
$R python tools/verify_objects.py --json results/verify.json
```

램프가 **0/9** 를 내면 뭔가 잘못된 것이다. 08-25 빌드 기준 참고값:
2-link 2/3 · 3-link 8/9 · 램프 6/9, 최소 간격 11.00 mm.

`$R` = `../robot_learning/scripts/run_drake_env.sh`.
**맨 python 으로 부르면 pydrake 가 조용히 깨진다.**

---

## 1. 먼저 읽을 것 — 이 셋이면 시작할 수 있다

| 파일 | 무엇 | 언제 |
| --- | --- | --- |
| **[TEAMMATE_CHECKLIST.md](TEAMMATE_CHECKLIST.md)** (212줄) | **할 일 목록.** 순서·통과 기준·★ 9개 | 제일 먼저. 이것만 따라가도 된다 |
| [docs/pivot_experiment_guide.html](docs/pivot_experiment_guide.html) (405줄) | 실험이 뭘 하는 건지, 왜 각 준비가 필요한지 | 처음 하는 사람이면 체크리스트보다 먼저 |
| [EXPERIMENT_READY.md](EXPERIMENT_READY.md) (359줄) | 무엇을 왜 고쳤는지 + 검증 숫자 | 체크리스트가 "왜?" 라고 느껴질 때 |

안내서는 브라우저로 연다: `xdg-open docs/pivot_experiment_guide.html`

---

## 2. 실행할 도구 — 순서대로

```
tools/make_convex.py       ①  물체 충돌 형상을 볼록 조각으로 (coacd 필요)
tools/verify_objects.py    ②  세 물체가 되는지 확인    ← 6/9 같은 숫자가 여기서
tools/preflight.py         ③  준비 점검 (실패 0 이어야 다음으로)
tools/import_easyhec.py    ④  EasyHeC Tc_c2b -> PIVOT X_WC
tools/angle_signs.py       ⑤  각도 부호·영점 + 실제 각도 오차

tools/make_part_legend.py  —  런처가 알아서 부른다 (부위 이름표 그림)
tools/grasp_measure.py     —  창 1 이 「파지 완료」에서 부른다
tools/pivot_session.py     —  통합 UI 내부용 (세션 폴더·단계 상태)
```

각 파일 **맨 위 docstring 에 왜 필요한지·어떻게 쓰는지**가 다 있다.
`--help` 로도 나온다.

---

## 3. 반드시 적용해야 하는 것 — MeshPCA 쪽

```
integration/foundationpose/make_desk_lamp_masks.py  ->  MeshPCA/foundationpose/
integration/meshpca/tare_real.py                    ->  체크리스트 5번을 먼저 읽어라
```

★ **이 둘을 빠뜨리면 창 2 가 여전히 죽는다.** 2026-09-02 에 SAM3 가 base 를
0 픽셀로 잡아 FoundationPose 가 시작조차 못 했고, 런처는 180 초 기다리다
종료했다.

`tare_real.py` 는 그쪽 PC 에 **세 번째 버전**이 있다(`joint_deg` 를 쓰는 것).
통째로 덮지 말고 `--manual` / `--verify` 부분만 떼어 붙여라.

```bash
grep -l joint_deg $(find ~ -name tare_real.py 2>/dev/null)
```

---

## 4. 자산

```
assets/final_objects/lamp/README.md                  배치 + 어느 빌드인지 확인법
assets/final_objects/lamp/collision_meshes/convex/   볼록 조각 15개 (09-01 빌드용)
```

큰 메시(visual 22 MB + collision 32 MB)는 저장소에 없다. 배달물에서 복사한다.

★ **조각과 메시가 같은 빌드인지 확인해라.** 08-25 빌드와 09-01 빌드는 관절
원점이 **50~90 mm** 다르다. 섞으면 조용히 틀린다.

2-link 와 3-link 는 `my_work/density_id_objects.py` 에 상수로 들어 있다.
자산 작업이 필요 없다.

---

## 5. 무엇이 바뀌었는지 볼 때

```bash
git diff theory-and-deploy..real-experiment-ready -- my_work/
```

| 파일 | 핵심 변경 |
| --- | --- |
| `my_work/robot_scene.py` | 물체별 `GRASP_LONG_AXIS_BY_OBJECT`, 책상 실측 로더, **측정 파지 훅** |
| `my_work/desk_lamp.py` | 볼록 분해 없으면 **예외**, pinch 보정 복원, GT 값 갱신 |
| `my_work/dual_view.py` | 실행 사슬 검증, home 경유 후퇴, 도착 자세 간격 확인 |
| `my_work/pivot_ui.py` | **새 통합 UI** (단계 0~5) |
| `my_work/mesh_props.py` | `.obj` 읽기 추가 |
| `my_work/hardware_real.py` | `RbpodoBackend` 구현 |
| `setup/launch_experiment.sh` | 부위 이름표 자동 생성, 수동 마스크 기본 |

주석에 **왜 그렇게 했는지**와 사고 기록(관통 41.2 mm 등)을 같이 남겼다.

---

## 6. 이미 있던 것 중 꼭 볼 것

| 파일 | 왜 |
| --- | --- |
| [my_work/NAMING.md](my_work/NAMING.md) | 부위 이름 대응. **모르고 팀원 산출물을 이으면 조용히 틀린다** |
| [AGENTS.md](AGENTS.md) | 저장소 규칙 (`run_drake_env.sh` 필수, Drake 1.54 고정 등) |
| [assets/final_objects/lamp/README.md](assets/final_objects/lamp/README.md) | 빌드가 섞이면 안 되는 이유 |

---

## 막히면

| 증상 | 먼저 볼 것 |
| --- | --- |
| "힌지·도달·충돌을 모두 통과하는 자세가 없다" | `tools/verify_objects.py` — 어느 쌍이 몇 mm 겹치는지 나온다 |
| 창 2 가 안 뜨고 3분 뒤 종료 | `/tmp/pivot_win2.log`. 마스크 실패면 `MANUAL_MASK=1` |
| 라운드가 중간에 날아감 | 이제 시작 자세를 경유한다. 그래도 나면 경로 로그 |
| 결과가 그럴듯한데 이상함 | `preflight` 의 **주의** 항목부터. 명목값으로 도는 중이다 |

**`preflight` 에서 주의가 뜨면 "일단 돌려보자"가 아니라 "결과를 믿을 수 없다"로
읽어라.** 이 실험에서 위험한 것은 프로그램이 멈추는 게 아니라, 에러 없이 틀린
답이 나오는 것이다.
