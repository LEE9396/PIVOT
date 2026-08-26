# figures — 무엇이 논문에 들어가고 무엇이 진단인가

두 종류가 섞여 있습니다. **논문·발표에 쓰는 것**과, `study_*.py` 가 돌면서
남기는 **진단 그림**입니다. 앞의 것만 손으로 다듬었습니다.

## 논문에 들어가는 것

| 파일 | 만드는 것 | 무엇을 보이나 | 자리 |
| --- | --- | --- | --- |
| `fig_nullspace` | `make_ppt_figures.py` | 총질량·무게중심이 같은 두 밀도는 센서에 똑같이 보인다. 접으면 2.1 mm 갈라진다 | **Fig.2** III-C |
| `fig_precision_rounds` | `make_precision_figures.py` | 하한은 각도와 무관해 평평하고, 실제 라운드만 내려온다 | **Fig.3** IV-D |
| `fig_precision_halfwidth` | 〃 | 목표선을 가로지르는 지점 = 그 물체에 필요한 각도 정밀도 (설계 차트) | **Fig.4** IV-D |
| `fig_column_angle` | `make_ppt_figures.py` | 부위 열이 나란해진다 47.0° → 0.5°. 비이웃 쌍도 같이 무너진다 | IV-D 또는 부록 |
| `fig_spread_source` | 〃 | 라운드 편차는 잡음이 만들지 물체가 만들지 않는다 | 부록·발표 |
| `nlink_shapes` | `render_nlink.py` | p=4,5,6 의 펼침/중간/접힘. **되말림 진단의 반증** | 부록·발표 |
| `nlink_final` | `study_scaling.py` | 라운드 vs p, 하한을 계단으로 겹침 | `fig_precision_rounds` 에 흡수됨. 1단 폭이 필요하면 이것 |
| `linkage_density_panel_{2,3}link` | `linkage_density_run.py` → `linkage_density_panel.py` | 힌지 무시 / 모형화 / 실제값 | B1 · 발표 |
| `desklamp_density_panel` | `desklamp_density_render.py` → `desklamp_density_panel.py` | 램프 **시뮬** 탐색 전 / 후 / 실제값 | IV-D |
| `study_criterion` | `study_criterion.py` | D / A / E-최적 비교 | B3 · 발표 |
| `grasp_check_3link` | `study_grasp.py` | 후보 자세의 파지 가능성 판정 | III-D 여유 시 |

```bash
cd ~/Desktop/PIVOT/my_work
R=../robot_learning/scripts/run_drake_env.sh

$R python figures/make_ppt_figures.py          # nullspace / column_angle / spread_source
$R python study_precision.py                   # -> figures/precision.json
$R python figures/make_precision_figures.py    # rounds / halfwidth
$R python figures/linkage_density_run.py       # -> linkage_density_data.json
$R python figures/linkage_density_panel.py     # -> png + pdf
```

램프 쪽은 `open3d` + `trimesh` 가 있어야 메시를 렌더합니다(지금 환경엔
없습니다). 2-link / 3-link 는 **직육면체**라 Drake FK 로 자세만 받아
matplotlib 로 직접 그리므로 추가 의존성이 없습니다.

## 진단 그림 — `study_*.py` 가 남기는 것

`study_tilt` `study_stopping` `study_tls` `study_continuous` `study_startpose`
`study_grasp` `convergence` `lamp_assembled`.

**논문용이 아닙니다.** 축 라벨과 여백을 다듬지 않았고, 주장의 근거로 쓸 때는
숫자만 뽑아 표로 옮깁니다.

## `_superseded/` — 옛 판본

```
   nlink_validation          힌지를 넣기 전. 실물보다 쉬운 문제를 재고 있었다
   nlink_t2 / nlink_v2_low   목표 반폭을 정하는 중간 판본
   nlink_seed1001            seed 하나짜리 렌더. nlink_shapes 로 대체
   study_scaling{,_fine,_budget}   nlink_final 과 precision 으로 대체
   density_id_objects        active 와 random 이 겹쳐 보여 주장을 깎는다. 쓰지 말 것
```

지웠다가 되살릴 일이 있을까 봐 남겨 둡니다. **인용하지 마세요.**

---

## 힌지 비교 — 이 저장소에서 가장 값비싼 교훈

실물에는 힌지(41 g)가 **실제로 붙어 있고** 저울에도 같이 올라갑니다.
달라지는 것은 추정기가 그걸 아느냐뿐입니다.

```
ignored   추정기 모형에 힌지가 없다. 저울 총무게에는 들어 있으므로
          그 몫이 갈 곳이 없어 부위 밀도로 흘러든다.
modelled  힌지를 부위 하나로 같이 푼다 (지금 사양).
```

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

**새로 만든 `fig_*` 그림은 turbo 를 쓰지 않습니다.**

```
밀도       크기(magnitude)  ->  단일 색상 순차 램프 (파랑 250~700 스텝)
부위 수 p  순서 있는 양      ->  같은 램프. 선 구분은 끝에 붙인 이름이 담당
가설 A/B   정체성(identity) ->  범주형 두 칸 (파랑 #2a78d6 / 주황 #eb6834)
목표선     기준             ->  주황. 데이터 색과 겹치지 않게
```

색만으로 식별하지 않도록 **모든 선에 이름을 직접 붙였습니다.**

기존 `linkage_density_panel_*` 과 램프 그림은 turbo 를 씁니다. 명도만이 아니라
색상으로도 차이가 보이게 하려는 것인데, 알아 둘 것이 있습니다.

- 밀도 정의역은 **로그**입니다. 3-link 는 700 ~ 5200 으로 7.4 배라, 선형으로
  깔면 5200 하나가 눈금을 다 먹고 나머지 셋이 아래 20 % 에 뭉칩니다.
- turbo 는 순차형 인코딩의 표준 권고(단일 색상, 밝기 단조)를 따르지
  않습니다. 3-link 의 `link1_elbow`(청록)와 `link2_tip`(남색)이 적록색맹
  기준 ΔE 6.0 으로 **하한 밴드**에 걸립니다. 이 그림에서는 스와치 옆에
  **부위 이름과 숫자가 항상 같이 있으므로** 색만으로 식별하지 않습니다.

## 지운 것

`desklamp_density_{before,after,gt,strip}` — 8/27 삭제. 앞의 셋은
`desklamp_density_panel.py` 의 **입력**이었으므로, 패널 PNG 는 남아 있지만
지금은 다시 만들 수 없습니다 (이 환경에 `open3d`·`trimesh` 가 없습니다).

다시 만들어야 하면 되살리세요.

```bash
git checkout 92e3c47 -- my_work/figures/desklamp_density_{before,after,gt}.png
```
