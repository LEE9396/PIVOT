# scratch — 한 번 쓰고 남겨 둔 것

**어느 것도 파이프라인이 import 하지 않습니다.** 지우면 파이프라인은 그대로
돕니다. 남겨 둔 이유는 그때 무엇을 어떻게 쟀는지가 여기 적혀 있어서입니다.

| 파일 | 무엇을 재려고 만들었나 |
| --- | --- |
| `_diag.py` | 램프가 왜 자주 중단되는지 원인을 층별로 분리 |
| `_edge.py` | 구동범위 가장자리에서 후보가 살아남는지 |
| `_cheap.py` | 후보 검사에서 무엇이 시간을 먹는지 |
| `_gcheck.py` | 그리퍼 죠와 물체가 실제로 닿는지 |
| `_vertex.py` | glTF 정점 색이 Meshcat 까지 가는지 |
| `_shot.py` | 화면을 파일로 떠서 눈으로 확인 |
| `_grasp_flow.py` | 수동 파지 흐름 — 실제 팔 자세를 읽는가, 파지가 이동보다 먼저인가 (가짜 driver 를 물려 검사) |
| `template.py` | 새 실험을 시작할 때 복사해 쓰는 빈 틀 |
| `adaptive_select.py` | "왜 매 라운드 같은 각도가 나오는가" 를 파고든 메모. 결론(정보행렬이 y 에 의존하지 않는다)은 `design_core.py` 와 `ALGORITHM.md` 6장에 반영돼 있다 |

평소처럼 `my_work/` 에서 돌리면 됩니다.

```bash
cd ~/Desktop/PIVOT/my_work
../robot_learning/scripts/run_drake_env.sh python scratch/_grasp_flow.py
```

각 파일 맨 위에 상위 폴더를 `sys.path` 에 넣는 세 줄이 들어 있습니다.
`PYTHONPATH` 로는 안 됩니다 — `run_drake_env.sh` 가 ROS 오염을 걷어내려고
그 변수를 지우기 때문입니다.
