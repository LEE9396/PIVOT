#!/usr/bin/env python3
"""여러 자세의 (중력 방향, 렌치) 로 공구 파라미터와 센서 보정을 회귀로 푼다.

    python tools/tool_param_regression.py calibration/ft_poses_empty.jsonl [더 많은 파일/타어 json ...] \\
        --write-tare calibration/aft_tare_current.json --write-correction calibration/ft_correction.json

모형 (정지, 중력만 — 교수님 노트 A(q)·φ = F_e 의 정지 특수형)

    F_read = S_f · ( b_f + W_tool ĝ + m_load g ĝ )                 ← S_f: 센서 힘 축별 배율·회전 (3x3)
    T_read = S_t · ( b_t + (m·r)_tool × g ĝ + m_load (r_load × g ĝ) )

미지수
    b_f(3) b_t(3) W_tool(1) (m·r)_tool(3)                 ← 공구 파라미터 10개 (GravityTare 가 쓰는 것)
    S_f(3x3) S_t(3x3)                                     ← 센서 보정 (알려진 추 데이터가 있을 때만 식별됨)
    r_load(3)                                             ← 추의 무게중심 (추가 데이터가 있을 때)

빈 손 데이터만 있으면 S 는 단위행렬로 두고 공구 10개만 푼다 (= 지금의 타어).
알려진 추(--load-kg>0) 데이터가 섞여 있으면 힘 배율까지 같이 풀린다: 추를 물렸을 때
힘 크기 변화가 m_load·g 여야 하기 때문이다. 자세가 많을수록(수십) 공분산이 줄고
한 자세의 케이블 당김 같은 이상치가 잔차로 드러난다.

출력: 파라미터, 표준편차, 조건수, 자세별 잔차(이상치 표시), 그리고 원하면
GravityTare 형식 타어 파일(보정 단위)과 ft_correction.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "my_work"))
G = 9.81


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def load_rows(paths):
    rows = []
    for p in paths:
        p = Path(p)
        if p.suffix == ".jsonl":
            for line in p.read_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        else:                                       # tare json (entries)
            d = json.loads(p.read_text())
            for e in d["entries"]:
                w = e.get("wrench_raw_uncorrected", e["wrench"])
                rows.append(dict(load_kg=0.0, joint_deg=e.get("joint_deg"),
                                 g_hat=e.get("achieved_g_hat", e["g_hat"]), wrench=w))
    for r in rows:
        g = np.asarray(r["g_hat"], float); r["g_hat"] = (g / np.linalg.norm(g)).tolist()
    return rows


def fit_static(rows, solve_scale=True):
    """빈 손 + 알려진 추 자세를 한 번에 최소제곱. 반환: dict."""
    Gm = np.array([r["g_hat"] for r in rows]); W = np.array([r["wrench"] for r in rows], float)
    F, T = W[:, :3], W[:, 3:]; m_load = np.array([float(r.get("load_kg", 0)) for r in rows])
    has_load = solve_scale and np.any(m_load > 0)
    n = len(rows)

    # 1) 힘: 자유 선형 F = b + M ĝ  (+ 추가 있으면 추 항: M_load ĝ 로 별도 응답행렬, 같은 S_f 를 공유)
    #    빈 손: F = b + S_f W ĝ ;  추: F = b + S_f (W + m g) ĝ  → 추 자세의 응답이 (W + m g)/W 배.
    #    S_f·W 를 M 으로 두고, 추 자세에서는 M·(1 + m g / W) — W 가 미지수라 비선형. 2단계로 푼다.
    def solve_force(W_guess):
        A, y = [], []
        for g, f, m in zip(Gm, F, m_load):
            k = 1.0 + (m * G / W_guess if W_guess else 0.0)
            # (M g)_i = Σ_j M_ij g_j  ->  행 i 의 열 3i+j 가 g_j : kron(I3, gᵀ). (kron(g, I) 는 Mᵀ g 가
            # 되어 회전 부호가 뒤집힌다 — 자체 테스트에서 C_f·S_f 가 6° 회전으로 드러났다.)
            A.append(np.hstack([np.eye(3), k * np.kron(np.eye(3), g[None, :])]))   # b(3) + M(9)
            y.append(f)
        A = np.vstack(A); y = np.concatenate(y); x, *_ = np.linalg.lstsq(A, y, rcond=None)
        b, M = x[:3], x[3:].reshape(3, 3)
        res = y - A @ x
        return b, M, res, A
    # 초기 W: 빈 손 자세만으로 M 을 맞춘 뒤 평균 gain. (k=1 로 전부 넣고 시작하면 추 하중이
    # M 에 흡수돼 W 가 발산한다 — 자체 테스트에서 추 자세 잔차 2 N 으로 드러났다.)
    empty = m_load <= 0
    A0 = np.vstack([np.hstack([np.eye(3), np.kron(np.eye(3), g[None, :])]) for g in Gm[empty]])
    x0, *_ = np.linalg.lstsq(A0, F[empty].reshape(-1), rcond=None)
    W_guess = float(np.linalg.svd(x0[3:].reshape(3, 3), compute_uv=False).mean())
    for _ in range(50):                                   # W 를 반복 갱신 (추 데이터 없으면 1회)
        b_f, M, res_f, A_f = solve_force(W_guess)
        U, S, Vt = np.linalg.svd(M); R = U @ Vt
        # 절대 배율: 추가 있으면 M 은 S_f·W 인데 S_f 의 크기(평균 배율)는 추 항에서 나온다:
        # 추 자세 잔차가 최소가 되는 W 를 찾는 것과 같다. 없으면 W = 평균 gain (S_f = 회전만).
        W_new = S.mean() if not has_load else None
        if has_load:
            # 추 자세만 골라 |F - b - M g| 대신 (F - b) 의 M^-1 사상 크기 = (W + m g)/W·|g| 로 W 추정
            mask = m_load > 0
            k_obs = np.array([np.linalg.norm(np.linalg.inv(M) @ (F[i] - b_f)) for i in np.where(mask)[0]])
            W_new = float(np.mean(m_load[mask] * G / np.maximum(k_obs - 1.0, 1e-6)))
        if abs(W_new - W_guess) < 1e-6:
            break
        W_guess = 0.5 * (W_guess + W_new)               # 감쇠로 안정화
    W_tool = W_guess
    S_f = M / W_tool                                      # 읽기 = S_f · 진짜  →  진짜 = S_f^-1 · 읽기
    C_f = np.linalg.inv(S_f)

    # 2) 토크: T = b_t + S_t [ (m r)_tool × g ĝ + m_load (r_load × g ĝ) ]
    #    S_t 는 힘의 회전만 준용하고 배율은 힘과 같다고 둔다 (추 레버가 미지수라 토크 배율은 따로 식별 안 됨).
    C_t = R.T / (W_tool / S.mean()) if False else np.linalg.inv(R * (np.linalg.det(S_f) ** (1/3)))
    Tc = (C_t @ T.T).T
    A, y = [], []
    for g, t, m in zip(Gm, Tc, m_load):
        blk = [np.eye(3), -skew(g) * G]                                        # b_t(3), (m r)_tool(3)
        if has_load:
            blk.append(-skew(g) * G * m)                                        # r_load(3)
        A.append(np.hstack(blk)); y.append(t)
    A = np.vstack(A); y = np.concatenate(y); x, *_ = np.linalg.lstsq(A, y, rcond=None)
    b_t, mr_tool = x[:3], x[3:6]
    r_load = x[6:9] if has_load else np.zeros(3)
    res_t = y - A @ x

    # 품질: 자세별 잔차, 파라미터 표준편차(잔차 분산 기반), 조건수
    Fc = (C_f @ F.T).T
    per_pose = []
    for i, (g, m) in enumerate(zip(Gm, m_load)):
        f_pred = C_f @ b_f + (W_tool + m * G) * g
        t_pred = b_t + G * np.cross(mr_tool + m * r_load, g)
        per_pose.append((np.linalg.norm(Fc[i] - f_pred), np.linalg.norm(Tc[i] - t_pred)))
    per_pose = np.array(per_pose)
    sig_f = np.sqrt(np.sum(per_pose[:, 0] ** 2) / max(3 * n - 12, 1))
    sig_t = np.sqrt(np.sum(per_pose[:, 1] ** 2) / max(3 * n - 9, 1))
    cov_t = sig_t ** 2 * np.linalg.pinv(A.T @ A)
    design = np.column_stack([np.ones(n), Gm]); cond = float(np.linalg.cond(design))
    return dict(n=n, has_load=bool(has_load), W_tool_n=float(W_tool), mass_tool_kg=float(W_tool / G),
                b_f=(C_f @ b_f).tolist(), b_t=b_t.tolist(), mr_tool=mr_tool.tolist(),
                com_tool_m=(mr_tool / (W_tool / G)).tolist(), r_load_m=r_load.tolist(),
                S_f=S_f.tolist(), C_f=C_f.tolist(), C_t=C_t.tolist(), rotation_deg=float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))),
                gains=S.tolist(), sigma_f=float(sig_f), sigma_t=float(sig_t),
                std_mr=np.sqrt(np.diag(cov_t))[3:6].tolist(), cond=cond, per_pose=per_pose.tolist())


def report(r, rows):
    print(f"자세 {r['n']}개 ({'알려진 추 포함' if r['has_load'] else '빈 손만'})   설계 조건수 {r['cond']:.1f}")
    print(f"  공구 무게 {r['W_tool_n']:.2f} N = {r['mass_tool_kg']:.3f} kg   무게중심 {np.round(1000*np.array(r['com_tool_m']),1).tolist()} mm"
          f"   (m·r 표준편차 {np.round(1000*np.array(r['std_mr']),2).tolist()} g·m)")
    print(f"  센서 힘 gain(읽기/진짜) {np.round(r['gains'],3).tolist()}  축 회전 {r['rotation_deg']:.2f}°"
          + ("" if r['has_load'] else "   [빈 손만이라 절대 배율은 1 로 가정 — 추 데이터를 섞으면 풀립니다]"))
    print(f"  힘 치우침 {np.round(r['b_f'],3).tolist()} N   토크 치우침 {np.round(r['b_t'],4).tolist()} N·m")
    print(f"  자세당 잔차 σ: 힘 {r['sigma_f']:.3f} N, 토크 {r['sigma_t']:.4f} N·m")
    pp = np.array(r["per_pose"]); bad = np.where((pp[:, 0] > 3 * r['sigma_f']) | (pp[:, 1] > 3 * r['sigma_t']))[0]
    for i in bad:
        print(f"    [이상치] 자세 {i}: 힘 {pp[i,0]:.3f} N 토크 {pp[i,1]:.4f}  관절 {np.round(rows[i].get('joint_deg') or [],1).tolist()}")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", type=Path)
    ap.add_argument("--write-tare", type=Path, default=None, help="GravityTare 형식(보정 단위)으로 저장")
    ap.add_argument("--write-correction", type=Path, default=None, help="ft_correction.json 으로 저장")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv[1:])
    if args.self_test:
        sys.exit(0 if self_test() else 1)
    rows = load_rows(args.inputs)
    if len(rows) < 6:
        sys.exit(f"자세가 {len(rows)}개뿐입니다 — 최소 6, 권장 20 이상")
    r = fit_static(rows); report(r, rows)
    if args.write_correction:
        C = np.zeros((6, 6)); C[:3, :3] = r["C_f"]; C[3:, 3:] = r["C_t"]
        args.write_correction.write_text(json.dumps(dict(matrix=C.tolist(), created=time.strftime("%Y-%m-%d"),
            frame="ft_mount", method="tools/tool_param_regression.py", n_poses=r["n"], has_known_load=r["has_load"],
            gains_read_over_true=r["gains"], rotation_deg=r["rotation_deg"]), indent=2) + "\n")
        print(f"  -> {args.write_correction}")
    if args.write_tare:
        C = np.zeros((6, 6)); C[:3, :3] = r["C_f"]; C[3:, 3:] = r["C_t"]
        entries = [dict(g_hat=row["g_hat"], achieved_g_hat=row["g_hat"], joint_deg=row.get("joint_deg"),
                        wrench=(C @ np.asarray(row["wrench"], float)).tolist(), wrench_raw_uncorrected=row["wrench"])
                   for row in rows if float(row.get("load_kg", 0)) == 0]
        payload = dict(entries=entries, wrench_frame="ft_mount", created_at_s=time.time(),
                       regression=dict({k: v for k, v in r.items() if k != "per_pose"}),
                       manual=dict(method="tools/ft_collect_poses.py freedrive", tool_mass_kg=r["mass_tool_kg"]))
        import tempfile, tare_check as tc
        with tempfile.NamedTemporaryFile("w", suffix=".json") as tmp:
            json.dump(payload, tmp); tmp.flush()
            passed, _ = tc.check(tmp.name, tool_kg=r["mass_tool_kg"])
        payload["measured"] = dict(passed=bool(passed), forced=False, override=tc.current_overrides())
        args.write_tare.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"  -> {args.write_tare}  passed={passed}")


def self_test(seed=0):
    """알려진 파라미터로 자세 30개를 만들어 되찾는지 본다 (배율·회전·추 포함)."""
    rng = np.random.default_rng(seed)
    W, mr = 6.9, np.array([0.002, -0.003, 0.046]); bf, bt = np.array([0.3, -24.0, 11.0]), np.array([-0.02, 0.05, 0.01])
    ang = np.radians(3.0); Rz = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    S_f = Rz @ np.diag([1.41, 1.44, 1.60]); S_t = Rz * 1.48
    rows = []
    for i in range(30):
        g = rng.normal(size=3); g /= np.linalg.norm(g); m = 1.332 if i % 3 == 0 else 0.0
        r_load = np.array([0.01, -0.02, 0.15])
        f = S_f @ (bf + (W + m * G) * g) + rng.normal(0, 0.05, 3)
        t = S_t @ (bt + G * np.cross(mr + m * r_load, g)) + rng.normal(0, 0.002, 3)
        rows.append(dict(load_kg=m, g_hat=g.tolist(), wrench=np.concatenate([f, t]).tolist(), joint_deg=[0] * 6))
    r = fit_static(rows); report(r, rows)
    ok = abs(r["mass_tool_kg"] - W / G) < 0.03 and np.allclose(r["com_tool_m"], mr / (W / G), atol=0.005)
    print("self-test", "OK" if ok else "FAIL", f"(공구 {r['mass_tool_kg']:.3f} vs {W/G:.3f} kg)")
    return ok


if __name__ == "__main__":
    main(sys.argv)
