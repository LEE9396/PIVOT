# figures — 문서가 근거로 참조하는 그림

`study_*.py` 가 남기는 그림과, 손으로 만든 비교 패널이 함께 있습니다.

## 부위별 밀도 비교 패널

| 그림 | 만드는 것 | 무엇을 보이나 |
| --- | --- | --- |
| `desklamp_density_panel.png` | `desklamp_density_render.py` → `desklamp_density_panel.py` | 램프: 탐색 **전 / 후 / 실제값** |
| `linkage_density_panel_2link.png` | `linkage_density_run.py` → `linkage_density_panel.py` | 2-link: 힌지 **무시 / 모형화 / 실제값** |
| `linkage_density_panel_3link.png` | 〃 | 3-link: 〃 |

```bash
cd ~/Desktop/PIVOT/my_work
R=../robot_learning/scripts/run_drake_env.sh
$R python figures/linkage_density_run.py      # 수치 -> linkage_density_data.json
$R python figures/linkage_density_panel.py    # 그림 -> png + pdf
```

램프 쪽은 `open3d` + `trimesh` 가 있어야 메시를 렌더합니다(지금 환경엔 없습니다).
2-link / 3-link 는 **직육면체**라 Drake FK 로 자세만 받아 matplotlib 로 직접
그리므로 추가 의존성이 없습니다.

### 무엇을 비교하나

실물에는 힌지(41 g)가 **실제로 붙어 있고** 저울에도 같이 올라갑니다.
달라지는 것은 추정기가 그걸 아느냐뿐입니다.

```
ignored   추정기 모형에 힌지가 없다. 저울 총무게에는 들어 있으므로
          그 몫이 갈 곳이 없어 부위 밀도로 흘러든다.
modelled  힌지를 부위 하나로 같이 푼다 (지금 사양).
```

두 경우 모두 **진리 plant 에는 힌지 질량이 들어 있습니다.** 그래야 공정합니다.

### 읽는 법 — 숫자보다 이 대비가 핵심

| | 부위 질량오차 (최대) | 알고리즘이 주장하는 반폭 |
| --- | ---: | ---: |
| 2-link 무시 | 7.7 % | ±0.48 % |
| 2-link 모형화 | 0.02 % | ±0.55 % |
| 3-link 무시 | **23.0 %** | **±0.38 %** |
| 3-link 모형화 | 0.01 % | ±1.05 % |

**힌지를 빼먹으면 3-link 가 23 % 틀리는데 알고리즘은 0.38 % 라고 말합니다.**
60 배 과신입니다. 이 오차는 잡음이 아니라 모형이 빠뜨린 것이라 라운드를
늘려도 안 없어집니다.

## 색에 대하여

`turbo` 를 씁니다. 명도만이 아니라 **색상으로도** 차이가 보이게 하려는
것이고, 램프 그림과 같은 규약입니다. 다만 알아 둘 것이 있습니다.

- 밀도 정의역은 **로그**입니다. 3-link 는 700 ~ 5200 으로 7.4 배라, 선형으로
  깔면 5200 하나가 눈금을 다 먹고 나머지 셋이 아래 20 % 에 뭉칩니다.
- turbo 는 순차형 인코딩의 표준 권고(단일 색상, 밝기 단조)를 따르지
  않습니다. 팔레트를 검사해 보면 3-link 의 `link1_elbow`(청록)와
  `link2_tip`(남색)이 적록색맹 기준 ΔE 6.0 으로 **하한 밴드**에 걸립니다.
  이 그림에서는 스와치 옆에 **부위 이름과 숫자가 항상 같이 있으므로**
  색만으로 식별하지 않습니다. 색은 렌더와 표를 잇는 보조 표식입니다.
- 단일 색상 순차 램프(viridis 등)로 바꾸면 그 경고는 사라지지만, 부위
  구분이 밝기 차이로만 남습니다. 지금은 구분 쪽을 택했습니다.
