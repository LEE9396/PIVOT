# PIVoT v2 초안 인계 — 자료가 도착하면 어디를 갈아끼우는가

작업 트리: `~/Desktop/PIVOT/paper_v2/`  ·  빌드: 아래 한 줄
```bash
cd ~/Desktop/PIVOT/paper_v2
PATH=~/miniconda3/envs/test/bin:$PATH tectonic -X compile main.tex --outdir build
```
(LaTeX 은 conda `test` 환경의 tectonic 을 씁니다. 한글 `kotex` 포함해 정상 동작합니다.)

---

## 1. 팀원 인계 대기 2건

### (a) Laptop mesh / URDF
| 갈아끼울 곳 | 무엇을 |
|---|---|
| `my_work/density_id_objects.py` | `OBJECTS` 에 laptop `ObjectSpec` 추가 |
| `my_work/exp_v2_main.py` | `OBJECTS` 리스트에 `"laptop"` 추가 후 재실행 |
| `my_work/exp_v2_asset.py`, `exp_v2_path.py`, `exp_v2_volume.py`, `exp_v2_conv.py` | 같은 방식으로 laptop 추가 |
| `paper_v2/tables/table3_main_v2.tex` | laptop 행의 `\Pending` 을 실측치로 |
| `paper_v2/4_exp.tex` | "density results on three" 문장을 four 로 수정 |
| `paper_v2/main.tex` 초록, `1_intro.tex` 기여 3 | `\rev{...laptop...}` 문구 제거 |

GT 는 `paper_v2/gt.json` 에 이미 들어 있습니다 (Display 354.10 cm3 / 379.25 g, Base 798.60 cm3 / 1137.75 g).

### (b) part별 PCA scaling 부피 정제 코드 + 부피 오차 실측
| 갈아끼울 곳 | 무엇을 |
|---|---|
| `paper_v2/tables/table2_volume_v2.tex` | 44칸 `\Pending` 을 실측 부피 오차로 |
| `paper_v2/4_exp.tex` | `\input{tables/table2_volume_v2}` 한 줄의 주석을 해제 (지금은 지면 때문에 꺼 둠) |
| `paper_v2/4_exp.tex` Sec. B | `\rev{Their per-part volume-error table is deferred...}` 문장 교체 |
| `paper_v2/3_method.tex` Sec. B | `\rev{[pending: refinement implementation from collaborator]}` 2곳 제거 |

**주의**: Fig. 4(부피 전파, `fig_volume.png`)는 **이 자료와 무관하게 이미 완성**돼 있습니다.
정제 코드가 없어도 "부피 오차가 밀도·관성·라운드 수에 어떻게 전파되는가"는 이미 정량화했습니다.
도착하는 자료는 "우리 정제가 그 오차를 얼마나 줄이는가"를 채웁니다.

---

## 2. 계산 중인 실험 (자동으로 채워지면 갱신)

`study_strategy.py`, `study_precision.py`, `study_criterion.py`, `study_centroid.py` 가
p=5 에서 계산 중입니다(문서화된 계산 "벽"). 끝나면:

| 파일 | 무엇을 |
|---|---|
| `paper_v2/4_exp.tex` L~218 | `\rev{The $P=5,6$ arms of the strategy comparison are still computing.}` 제거하고 수치 반영 |

로그: `my_work/logs_v2/*.log`, JSON: `my_work/figures/v2/*.json`.
**`study_baselines.py` 는 p=2..6 전부 완료**되어 이미 본문에 반영돼 있습니다.

---

## 3. 실물 로봇
`REAL_ROBOT_TODO.md` 참조. 순서(2link → 3link → 램프 → 노트북)가 중요합니다.

---

## 4. 확인 요망 (미해결)
`DATA_NOTE.md` — `gt_list.md` 의 램프 3개 part 와 노트북 Base 의 **밀도 열이 부피 열과 불일치**합니다.
그 밀도들은 v1 의 옛 부피로 계산하면 정확히 맞습니다. 현재는 질량·부피를 측정량으로 보고
`rho = M/V` 로 재계산해 썼습니다 (`gt.json`). 어느 쪽이 최신인지 확인해 주십시오.

---

## 5. 새로 만든 실험 스크립트 (전부 `my_work/`, 기존 파일 미수정)

| 스크립트 | 산출 | 논문 위치 |
|---|---|---|
| `exp_v2_main.py` | `figures/v2/main_real.json` | Table III (메인) |
| `exp_v2_path.py` | `path_compare.json`, `fig_path.png` | Table IV, Fig. 6 |
| `exp_v2_asset.py` | `asset_validation.json`, `fig_asset.png` | Table V, Fig. 7 |
| `exp_v2_volume.py` | `volume_propagation.json`, `fig_volume.png` | Fig. 4 |
| `exp_v2_conv.py` | `convergence.json`, `fig_conv.png` | Fig. 5 |
| `replot_v2.py` | 그림 재생성 (실험 재실행 없이) | — |

전부 `../robot_learning/scripts/run_drake_env.sh python <script>` 로 `my_work/` 에서 실행합니다.

**수정한 기존 파일은 하나뿐**: `desk_lamp.py` 의 `GROUND_TRUTH` 를 gt_list.md 값으로 갱신
(원본은 `desk_lamp.py.v1bak`).

---

## 6. 문서 지도
| 파일 | 내용 |
|---|---|
| `RESULTS.md` | 모든 실험 결과 + **정직하게 써야 할 주의사항** (집필 원자료) |
| `WRITING_BRIEF.md` | 집필 브리프 — 과학적 사실, 문헌 확인 결과, 표기 규칙 |
| `COMPRESSION_RULES.md` | 분량 압축 규칙 + **삭제 금지 목록** |
| `CUTS.md` | 압축 과정에서 덜어낸 내용 (복원용) |
| `DATA_NOTE.md` | GT 정합성 문제 |
| `REAL_ROBOT_TODO.md` | 실물 실험 계획 |
| `gt.json` | GT 정본 |

---

## 7. 분량 — 9페이지, 1페이지 초과 (결정 필요)

초안은 **9페이지**입니다. ICRA 한계는 6페이지, 초과 게재료를 내면 8페이지입니다.
14페이지에서 시작해 산문을 8,461 → 5,081 단어로 줄이고 서지를 38 → 23개로 정리했습니다.

**9에서 8로 가려면 아래 중 하나를 선택해야 하며, 이것은 내용 판단이라 남겨 둡니다.**

| 안 | 무엇을 뺄까 | 비용 |
|---|---|---|
| **A** | 자리표시자 그림 3개 중 하나 (Fig. 1 teaser / Fig. 2 pipeline / Fig. 3 setup). Fig. 3 이 가장 덜 아깝다 — 물체 사양은 이미 캡션에 흡수됐다 | 약 0.2쪽. 리뷰어는 setup 사진을 기대하지만 필수는 아니다 |
| **B** | 실험 한 절을 통째로. **Sec. IV-B(부피)** 가 후보 — 지금 Table 2 가 협업자 대기라 비어 있고, Fig. 4 만 남아 있다 | 약 0.5쪽. 다만 제목의 "Volume-Aware" 근거가 약해진다 |
| **C** | Method 의 정리 9개 중 2~3개를 진술만 남기고 본문에서 각주로 | 약 0.4쪽. 이론 기여가 흐려진다 |
| **D** | 8페이지를 포기하고 **초과 게재료 8페이지 + 내용 1쪽 추가 압축**을 편집 단계로 미룬다 | 0 |

**권장: A + C 조합.** 실험을 빼지 않고 8페이지에 들어갈 수 있습니다.
자료가 도착하면(laptop, 부피 정제) 분량이 다시 늘어나므로, 그때 다시 재야 합니다.

압축 과정에서 덜어낸 내용은 전부 `CUTS.md` 에 있어 복원할 수 있습니다.

### 압축하며 지킨 것 (되돌리지 마십시오)
정직성 단서 18개 항목을 삭제 금지 목록(`COMPRESSION_RULES.md`)으로 고정하고 전부 보존했습니다.
특히 다음은 논문의 방어선입니다.
- 밀도 오차 = 질량 오차 (1.7e-16) 라는 사실과 한 열로만 보고한다는 것
- 2link 에서 ours ≡ single 이라는 것 (승리로 제시하지 않음)
- single 의 실패가 점오차가 아니라 반폭에 있다는 것
- 질량이 부피 오차에 **엄밀히** 불변하지 않다는 것, 부피 오차가 라운드를 늘린다는 것
- 관절한계 위반 열이 해석적으로 무의미하다는 것
- 62 mm 일화를 재현했다고 쓰지 않은 것
- Exp E 가 시뮬-시뮬 대리 검증이라는 것, 2link RELEASE 가 밀도에 둔감하다는 것
- 관성 텐서는 준정적으로 식별 불가이며 유도량이라는 것
- Kumar et al. 2019 을 구분해 인용한 문장
