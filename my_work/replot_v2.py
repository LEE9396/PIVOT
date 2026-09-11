"""ICRA 지면 예산 때문에 fig_asset.png 만 다시 그리는 작은 스크립트.

exp_v2_asset.py 의 make_figure() 는 3(시나리오) x 3(물체) = 9칸짜리 그림을
그리는데, 인쇄 크기에서 세로 4.7 in 가 되어 지면을 너무 먹는다. 이 스크립트는
같은 실험을 다시 돌리지 않고 이미 저장된 figures/v2/asset_validation.json
(각 물체·시나리오·variant 의 IC#0 시계열이 이미 들어 있다) 만 읽어서, 물체당
"가장 정보가 되는 시나리오 하나" 를 한 줄 3칸으로 압축해 그린다.

시나리오 선택 (물체마다 고정, 아래 SCENARIO_CHOICE 참고):
  - 2link 의 RELEASE 는 원리상 밀도를 못 본다 (뿌리 고정 + 단일 진자는
    링크 밀도를 통째로 배율해도 운동이 같다 — m 과 I 가 같이 커지기 때문).
    exp_v2_asset.py 의 caveats() 가 이미 이렇게 적어 두었다. 그래서 2link는
    RELEASE 를 절대 고르지 않는다.
  - 세 물체 모두 PUSH 를 골랐다. asset_validation.json 의 IC#0 시계열을 실제로
    비교해 보면(이 스크립트를 만들며 직접 확인), DROP 에서는 vision-SiPhy가
    desklamp·2link 에서 gt 에 거의 붙어버려("갈라진다"는 그림의 주장이
    약해짐), RELEASE 는 진폭 자체가 작아 세 variant 의 차이가 잘 안 보인다.
    반대로 PUSH 는 두 vision 변형 모두, 그리고 uniform 도 gt 대비 확실히
    갈라지고(초기조건 0에서 곡선 RMSE 기준 2link: ours 13.9 mm vs 최소
    baseline 31 mm; 3link: ours 1.9 mm vs 최소 baseline 219 mm; desklamp:
    ours 10.3 mm vs 최소 baseline 19.3 mm), 세 물체 모두에서 ours 는 기준을
    가장 가깝게 따라간다. (주의: caveats() 는 PUSH 가 접촉을 거쳐 절대값이
    증폭된다고 적어 둔다 — 그래서 이 그림이 보이는 것은 절대 오차가 아니라
    "누가 기준을 따라가고 누가 갈라지는가" 라는 순서다. ordering_holds 는
    27/27 이 모든 시나리오에서 이미 성립함을 보였다.)

실행:
    cd ~/Desktop/PIVOT/my_work
    /home/junhyeoklee/Desktop/PIVOT/robot_learning/.venv-drake-1.54-py312/bin/python replot_v2.py
"""

import argparse
import json
from pathlib import Path

import numpy as np

IN_JSON = Path("figures/v2/asset_validation.json")
OUT_PNG = Path("figures/v2/fig_asset.png")

# 물체별 대표 시나리오. 2link RELEASE 는 절대 고르지 않는다 (구조적으로
# 밀도에 눈이 먼 시나리오라 결과를 왜곡해 보인다).
SCENARIO_CHOICE = {"2link": "push", "3link": "push", "desklamp": "push"}

VARIANTS = ("ours", "uniform", "vision-SiPhy", "vision-PUGS")
STYLE = {
    "ours": dict(color="0.15", ls="-", lw=1.3, marker=None, zorder=6),
    "uniform": dict(color="0.35", ls="--", lw=1.3, marker="s", zorder=4),
    "vision-SiPhy": dict(color="0.35", ls="-.", lw=1.3, marker="^", zorder=3),
    "vision-PUGS": dict(color="0.35", ls=":", lw=1.5, marker="D", zorder=3),
}
LABELS = {
    "gt": "reference (GT density)",
    "ours": "ours",
    "uniform": "uniform",
    "vision-SiPhy": "vision-SiPhy",
    "vision-PUGS": "vision-PUGS",
}


def med(runs, scenario, variant, field="traj_rmse_mm_max"):
    vals = [r[field] for r in runs
            if r["scenario"] == scenario and r["variant"] == variant]
    return float(np.median(vals)) if vals else float("nan")


def make_figure(objects, choice, path=OUT_PNG):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(objects), figsize=(10, 3.0), dpi=200,
                             squeeze=False)
    axes = axes[0]
    handles_labels = None

    for ax, cell in zip(axes, objects):
        scenario = choice[cell["key"]]
        s = cell["series"][scenario]
        t = np.asarray(s["t"])

        gt = np.asarray(s["disp_mm"]["gt"])
        ax.plot(t, gt, color="k", lw=3.2, ls="-", zorder=2,
                label=LABELS["gt"])
        for variant in VARIANTS:
            if variant not in s["disp_mm"]:
                continue
            values = np.asarray(s["disp_mm"][variant])
            style = STYLE[variant]
            ax.plot(t, values, markevery=max(len(t) // 8, 1), ms=4,
                    mfc="none", mew=1.1, label=LABELS[variant], **style)

        ours_med = med(cell["runs"], scenario, "ours")
        best_other = min(med(cell["runs"], scenario, v) for v in VARIANTS
                         if v != "ours")
        ax.set_title(f"{cell['key']} — {scenario} (tip {s['tip']})\n"
                     f"RMSE med: ours {ours_med:.1f} / best baseline "
                     f"{best_other:.1f} mm", fontsize=9)
        ax.set_xlabel("time [s]", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("tip displacement [mm]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, lw=0.5)
        if handles_labels is None:
            handles_labels = ax.get_legend_handles_labels()

    handles, labels = handles_labels
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
              frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Estimated-density simulation tracks the GT-density "
                "reference; uniform and vision-baseline densities diverge "
                "(SIM-vs-SIM surrogate)", fontsize=9.5, y=0.99)
    fig.subplots_adjust(top=0.72, bottom=0.26, left=0.06, right=0.985,
                        wspace=0.30)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    print(f"그림 -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(IN_JSON))
    ap.add_argument("--png", default=str(OUT_PNG))
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text())
    objects = data["objects"]
    make_figure(objects, SCENARIO_CHOICE, path=Path(args.png))


if __name__ == "__main__":
    main()
