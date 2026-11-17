"""빈 그리퍼 정지 기록의 시간 변화와 평균 잡음을 분리해 보고한다. 교정값은 바꾸지 않는다."""

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(report):
    if report.get("status") != "recorded" or report.get("load_state") != "empty":
        raise ValueError("완료된 빈 그리퍼 기록이 필요합니다")
    blocks = report["blocks"]
    t = np.concatenate([b["sample_time_s"] for b in blocks]).astype(float)
    y = np.vstack([b["wrench_samples"] for b in blocks])
    q = np.vstack([b["controller_joint_deg"] for b in blocks])
    if (len(t) < 2 or y.shape != (len(t), 6) or q.shape != (len(t), 6)
            or not all(np.all(np.isfinite(x)) for x in (t, y, q))
            or np.any(np.diff(t) <= 0)):
        raise ValueError("유효한 시간순 6축 원시 기록과 관절각이 필요합니다")
    t -= t[0]
    motion = float(np.max(np.abs((q - q[0] + 180.) % 360. - 180.)))
    if motion > .1:
        raise ValueError("정지 기록 중 자세가 0.1도 이상 바뀌었습니다")
    first = y[t < 60].mean(axis=0)
    last = y[t >= t[-1] - 60].mean(axis=0)
    means = np.asarray([b["wrench_mean"] for b in blocks])
    times = np.array([np.mean(b["sample_time_s"]) for b in blocks])
    slope = np.linalg.lstsq(np.column_stack([np.ones(len(times)),
                            (times - times[0]) / 60]), means, rcond=None)[0][1]
    bins = t.astype(int) // 60
    minute = [dict(start_s=int(k * 60), samples=int(np.sum(bins == k)),
                   mean=y[bins == k].mean(axis=0).tolist()) for k in np.unique(bins)]
    # 현재 50 Hz / 1000샘플 측정에 대응하는 20초 평균의 실제 산포도 남긴다.
    hold_means = np.array([y[(t >= k*20) & (t < (k+1)*20)].mean(axis=0)
                          for k in range(int(t[-1] // 20))])
    return dict(source="analyze_ft_drift", calibration_valid=False,
                duration_s=float(t[-1]), collection_elapsed_s=report.get("record_elapsed_s"),
                requested_10min_completed=bool(report.get("record_elapsed_s", t[-1]) >= 600),
                sample_count=len(t), block_count=len(blocks), max_joint_change_deg=motion,
                mean=y.mean(axis=0).tolist(), sample_std=y.std(axis=0, ddof=1).tolist(),
                first_60s_mean=first.tolist(), last_60s_mean=last.tolist(),
                last_minus_first_60s=(last-first).tolist(),
                force_endpoint_change_n=float(np.linalg.norm((last-first)[:3])),
                torque_endpoint_change_nm=float(np.linalg.norm((last-first)[3:])),
                fitted_slope_per_minute=slope.tolist(),
                block_mean_span_per_axis=np.ptp(means, axis=0).tolist(),
                block_mean_covariance=np.cov(means, rowvar=False).tolist(),
                hold_20s_mean_std=(hold_means.std(axis=0, ddof=1).tolist()
                                  if len(hold_means) >= 2 else None),
                hold_20s_mean_covariance=(np.cov(hold_means, rowvar=False).tolist()
                                         if len(hold_means) >= 2 else None),
                max_sample_gap_s=float(np.max(np.diff(t))),
                minute_means=minute,
                temperature_recorded=False, thermal_cause_confirmed=False,
                notes="구간 평균의 공분산은 기술통계다. 독립 표본이나 밀도의 1σ로 간주하지 않는다.",
                robot_commands_sent=report.get("robot_commands_sent"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("record", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--plot", type=Path)
    args = ap.parse_args()
    report = json.loads(args.record.read_text())
    result = analyze(report)
    result["record_file"] = str(args.record.resolve())
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        blocks = report["blocks"]
        t = np.array([np.mean(b["sample_time_s"]) for b in blocks])
        y = np.array([b["wrench_mean"] for b in blocks])
        fig, axes = plt.subplots(3, 2, figsize=(11, 7), sharex=True)
        for i, ax in enumerate(axes.T.flat):
            ax.plot((t-t[0])/60, y[:, i], lw=.9)
            ax.set_ylabel(["Fx [N]", "Fy [N]", "Fz [N]",
                           "Tx [Nm]", "Ty [Nm]", "Tz [Nm]"][i])
            ax.grid(alpha=.25)
        for ax in axes[-1]:
            ax.set_xlabel("Time [min]")
        fig.suptitle("Empty gripper: uncorrected block means")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        plt.close(fig)
    print(json.dumps({k: result[k] for k in ("duration_s", "sample_count",
        "last_minus_first_60s", "force_endpoint_change_n", "max_joint_change_deg")}, indent=2))


if __name__ == "__main__":
    main()
