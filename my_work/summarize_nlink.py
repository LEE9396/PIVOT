"""exp_nlink_density.py 의 CSV 에서 표 A / 표 B / L1-L6 비교를 만든다.

지시문 6절의 두 표만 만든다. 그래프는 만들지 않는다.

실행:
  cd my_work
  ../robot_learning/scripts/run_drake_env.sh python -u summarize_nlink.py
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

CONDS = ("C0", "C1", "C2", "C3")
COND_LABEL = {"C0": "기준 / 기준", "C1": "상향 / 기준",
              "C2": "기준 / 상향", "C3": "상향 / 상향"}
PAPER_CONDS = ("C0", "C1", "C2")     # 표 B 는 C3 미게재
PAPER_LINKS = (2, 3, 6)


def load(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def cell(rows):
    """trial 평균 ± 표준편차 (소수점 1자리). 수렴 실패도 포함해 센다."""
    if not rows:
        return "—"
    v = np.array([float(r["mean_err_pct"]) for r in rows])
    return f"{v.mean():.1f} ± {v.std(ddof=0):.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/nlink_density_results.csv")
    ap.add_argument("--out", default="results/nlink_summary.md")
    ap.add_argument("--notes", default="results/nlink_notes.md",
                    help="손으로 쓴 보고 문장. 있으면 요약 끝에 이어 붙인다.")
    args = ap.parse_args()

    rows = load(args.csv)
    by = defaultdict(list)
    for r in rows:
        by[(int(r["n_links"]), r["cond_id"])].append(r)
    links = sorted({int(r["n_links"]) for r in rows})
    conds = [c for c in CONDS if any(k[1] == c for k in by)]
    n_trials = max(len(v) for v in by.values())

    out = []
    w = out.append
    w("# N-link 밀도추정 시뮬레이션 — 집계 결과\n")
    w(f"원본: `{os.path.basename(args.csv)}` — "
      f"{len(links)}(N) x {len(conds)}(조건) x {n_trials}(trial) = {len(rows)} trial\n")
    w("셀은 trial 별 `mean_err_pct`(링크 평균 |밀도오차|)의 **평균 ± 표준편차 [%]** 이다. "
      "수렴 실패한 trial 도 버리지 않고 포함했다.\n")

    # ---------------- 표 A ----------------
    w("\n## 표 A — 전체 (내부 확인용)\n")
    w("| 조건 | " + " | ".join(f"{n} links" for n in links) + " |")
    w("|---|" + "---|" * len(links))
    for c in conds:
        w(f"| {c} | " + " | ".join(cell(by[(n, c)]) for n in links) + " |")

    # ---------------- 표 B ----------------
    pl = [n for n in PAPER_LINKS if n in links]
    w("\n## 표 B — 논문 게재용\n")
    w("| Volume err. / Angle err. | " + " | ".join(f"{n} links" for n in pl) + " |")
    w("|---|" + "---|" * len(pl))
    for c in PAPER_CONDS:
        if c not in conds:
            continue
        w(f"| {COND_LABEL[c]} | " + " | ".join(cell(by[(n, c)]) for n in pl) + " |")
    w("\n기준 = volume 1 %, angle 1 % / 상향 = 5 %. C3(상향·상향)는 지시대로 미게재.\n")

    # ---------------- L1 vs L6 ----------------
    w("\n## 오차 전파 확인 — 6링크 C0 의 첫 링크 / 마지막 링크\n")
    six = by.get((6, "C0"), [])
    if six:
        w("| seed | err L1 [%] | err L6 [%] |")
        w("|---|---|---|")
        l1 = np.array([float(r["err_pct_L1"]) for r in six])
        l6 = np.array([float(r["err_pct_L6"]) for r in six])
        for r, a, b in zip(six, l1, l6):
            w(f"| {r['seed']} | {a:.1f} | {b:.1f} |")
        w(f"| **평균** | **{l1.mean():.1f} ± {l1.std(ddof=0):.1f}** "
          f"| **{l6.mean():.1f} ± {l6.std(ddof=0):.1f}** |")
        w("")
        # 링크 위치별 평균도 같이 둔다 (전파가 단조인지 보려면 중간이 필요하다).
        prof = [np.mean([float(r[f"err_pct_L{i}"]) for r in six])
                for i in range(1, 7)]
        w("6링크 C0 링크별 평균 오차율 [%]: "
          + ", ".join(f"L{i + 1} {v:.1f}" for i, v in enumerate(prof)))
    else:
        w("(6링크 C0 데이터 없음)")

    # ---------------- 부수 지표 ----------------
    w("\n## 부수 지표\n")
    w("| 조건 | " + " | ".join(f"{n} links" for n in links) + " |")
    w("|---|" + "---|" * len(links))
    for c in conds:
        cells = []
        for n in links:
            g = by[(n, c)]
            if not g:
                cells.append("—")
                continue
            ok = sum(r["converged"] == "True" for r in g)
            cfg = np.mean([int(r["n_configs"]) for r in g])
            cells.append(f"{ok}/{len(g)} 수렴, config {cfg:.1f}")
        w(f"| {c} | " + " | ".join(cells) + " |")
    w("")
    w("| 조건 | " + " | ".join(f"{n} links" for n in links) + " |")
    w("|---|" + "---|" * len(links))
    for c in conds:
        cells = []
        for n in links:
            g = by[(n, c)]
            u = np.mean([float(r["final_uncertainty"]) for r in g]) if g else np.nan
            cells.append("—" if not g else f"반폭 {u:.1f}%")
        w(f"| {c} | " + " | ".join(cells) + " |")

    # ---------------- 수렴 실패 ----------------
    fails = [r for r in rows if r["converged"] != "True"]
    w("\n## 수렴 실패 목록\n")
    if not fails:
        w("없음 — 100 trial 전부 정지 조건에 도달했다.")
    else:
        w(f"{len(fails)} / {len(rows)} trial.\n")
        w("| N | 조건 | seed | config | 종료 반폭 [%] | 목표 반폭 [%] | 평균오차 [%] |")
        w("|---|---|---|---|---|---|---|")
        for r in fails:
            n = int(r["n_links"])
            w(f"| {n} | {r['cond_id']} | {r['seed']} | {r['n_configs']} | "
              f"{float(r['final_uncertainty']):.1f} | {0.5 * n:.1f} | "
              f"{float(r['mean_err_pct']):.1f} |")

    # ---------------- 보고용 숫자 ----------------
    w("\n## 보고 문장에 쓸 숫자\n")
    for c in conds:
        seq = [f"{np.mean([float(r['mean_err_pct']) for r in by[(n, c)]]):.1f}"
               for n in links if by[(n, c)]]
        w(f"- {c} 링크수별 평균오차: " + " -> ".join(seq))
    if all((n, "C1") in by and (n, "C2") in by for n in links):
        ratios = []
        for n in links:
            c1 = np.mean([float(r["mean_err_pct"]) for r in by[(n, "C1")]])
            c2 = np.mean([float(r["mean_err_pct"]) for r in by[(n, "C2")]])
            ratios.append(f"N={n}: {c1 / max(c2, 1e-9):.2f}x")
        w("- C1(volume 상향) / C2(angle 상향) 비율: " + ", ".join(ratios))
    tot = sum(float(r["wall_time_sec"]) for r in rows)
    w(f"- 총 계산 시간 {tot:.0f}초 ({tot / 60:.1f}분), trial 평균 {tot / len(rows):.1f}초")

    # 지시문 7절의 서술은 해석이라 자동 생성하지 않는다. 손으로 쓴 것을
    # 별도 파일에 두고 여기 이어 붙인다 — 이 스크립트를 다시 돌려도 안 지워진다.
    if os.path.exists(args.notes):
        with open(args.notes) as fh:
            w("\n" + fh.read().rstrip())
    else:
        w(f"\n(보고 문장 없음 — `{args.notes}` 에 작성하면 여기 붙는다)")

    text = "\n".join(out) + "\n"
    with open(args.out, "w") as fh:
        fh.write(text)
    print(text)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
