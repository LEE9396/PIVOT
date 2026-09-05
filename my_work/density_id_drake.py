"""
Drake validation of measurement-grounded part-wise density identification
with active joint-configuration selection (D-optimality).

Scenario
--------
A 5-part "desk lamp" articulated object is grasped at its base. The wrist
F/T sensor frame S coincides with the world frame. Drake provides:
  (1) ground truth : total mass and CoM of the true-density plant
                     -> synthetic noisy static wrench measurements,
  (2) kinematics   : body poses of a unit-density twin plant
                     -> regressor columns for the estimator.
The estimator never sees the true densities; it only uses part volumes,
part-frame centroids, forward kinematics, and the noisy wrenches.

Pipeline stages (matching the paper's method section)
  S1 geometry preprocessing  -> volumes V_i, centroids (at body frame origin)
  S2 linear wrench model     -> y = A(theta, g_hat) rho + eps
  S3 candidate generation    -> joint-limit grid (+ feasibility hook)
  S4 information-gain score  -> Delta I(theta) = 1/2 logdet(I + R^-1/2 A S A^T R^-1/2)
  S5 orientation set         -> 3 near-orthogonal gravity directions in S
  S6 Bayesian update + box-constrained MAP + convergence / identifiability

Run:  python3 density_id_drake.py
"""

import numpy as np

from pydrake.geometry import Box
from pydrake.math import RigidTransform
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.multibody.tree import (
    FixedOffsetFrame,
    RevoluteJoint,
    SpatialInertia,
    UnitInertia,
)

try:
    from scipy.optimize import lsq_linear

    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

G_ACC = 9.81
RNG = np.random.default_rng(0)

# 센서가 어느 쪽을 읽는가. **이 저장소에서 힘 부호를 정하는 유일한 자리다.**
#
#   +1.0  센서가 물체가 아래로 당기는 힘을 읽는다   f = +m g ghat
#   -1.0  센서가 손목이 떠받치는 힘을 읽는다        f = -m g ghat
#
# 시뮬레이션에서는 measure() 가 만들고 regressor() 가 소비하므로 어느 쪽이든
# 상쇄되어 안 드러난다. 실물에서는 드러난다 — 반대로 잡으면 최소제곱이 밀도를
# 음수로 밀고, RHO_BOUNDS 하한(50)에 붙어 버린다. 실제로 그렇게 실패한 세션이
# 있다 (experiments/session_20260904_1736: 부위 3개 중 2개가 정확히 50).
#
# 실물이 어느 쪽인지는 **영점 조정 데이터로 확인한다.** 그리퍼 질량을 이미
# 아니까 추가 장비가 필요 없다. my_work/tare_check.py 가 그 검산을 한다.
# 값을 바꿀 때는 여기만 바꾼다. 읽는 쪽에서 또 뒤집으면 원위치가 된다.
FORCE_SIGN = +1.0

# ---------------------------------------------------------------------------
# Object definition: (name, box dims lx,ly,lz [m], true density [kg/m^3])
# Body frame of every part sits at its box centroid, so c_bar_local = 0 and
# the part centroid in sensor frame is simply the body-frame origin.
# ---------------------------------------------------------------------------
PARTS = [
    ("base",          (0.12, 0.12, 0.03), 2700.0),   # aluminum, grasped
    ("counterweight", (0.04, 0.04, 0.04), 7800.0),   # steel, welded to base
    ("lower_arm",     (0.03, 0.03, 0.25), 1200.0),   # ABS
    ("upper_arm",     (0.025, 0.025, 0.20),  900.0), # PP
    ("head",          (0.08, 0.05, 0.05), 4500.0),   # dense lamp head
]
JOINT_LIMITS = [(-1.6, 1.6), (-2.6, 2.6)]            # J1 (base->lower), J2

TRUE_RHO = np.array([p[2] for p in PARTS])
VOLUMES = np.array([np.prod(p[1]) for p in PARTS])
P = len(PARTS)

# F/T sensor noise (per axis, 1 sigma) — AFT200-class figures
SIGMA_F, SIGMA_T = 0.10, 0.003          # [N], [N m]
R_EPS_DIAG = np.array([SIGMA_F**2] * 3 + [SIGMA_T**2] * 3)

# Prior: uniform-density heuristic, deliberately wrong and broad
MU0 = np.full(P, 1000.0)
SIGMA0 = np.diag(np.full(P, 3000.0**2))
RHO_BOUNDS = (50.0, 20000.0)

# S5: gravity directions in sensor frame (robot reorients its wrist);
# three orthogonal directions maximize sigma_min of the stacked [g]_x blocks.
G_DIRS = [np.array(v) for v in ([0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])]


# ---------------------------------------------------------------------------
# Plant construction (used twice: true densities, and unit-density kinematics)
# ---------------------------------------------------------------------------
PART_COLORS = {
    "base": [0.55, 0.57, 0.60, 1.0],
    "counterweight": [0.25, 0.25, 0.28, 1.0],
    "lower_arm": [0.20, 0.55, 0.85, 1.0],
    "upper_arm": [0.35, 0.75, 0.55, 1.0],
    "head": [0.90, 0.60, 0.15, 1.0],
}


def build_lamp_plant(densities, builder=None):
    """Build the lamp. Numerics are unchanged; passing `builder` additionally
    registers box visuals through a SceneGraph so the object can be rendered."""
    if builder is None:
        plant = MultibodyPlant(time_step=0.0)
        scene_graph = None
    else:
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    bodies = {}
    for (name, dims, _), rho in zip(PARTS, densities):
        mass = rho * np.prod(dims)
        spatial = SpatialInertia(
            mass, np.zeros(3), UnitInertia.SolidBox(*dims)
        )
        bodies[name] = plant.AddRigidBody(name, spatial)
        if scene_graph is not None:
            plant.RegisterVisualGeometry(
                bodies[name],
                RigidTransform(),
                Box(*dims),
                f"{name}_visual",
                PART_COLORS[name],
            )

    def offset_frame(fname, body, xyz):
        return plant.AddFrame(
            FixedOffsetFrame(fname, body.body_frame(), RigidTransform(np.array(xyz)))
        )

    # base welded to sensor/world frame; box center 15 mm above grasp point
    plant.WeldFrames(
        plant.world_frame(), bodies["base"].body_frame(), RigidTransform([0, 0, 0.015])
    )
    # counterweight welded to base (offset in +x)
    plant.WeldFrames(
        offset_frame("cw_mount", bodies["base"], [0.05, 0.0, 0.020]),
        bodies["counterweight"].body_frame(),
    )
    # J1: revolute about y at top of base
    plant.AddJoint(
        RevoluteJoint(
            "J1",
            offset_frame("j1_parent", bodies["base"], [0.0, 0.0, 0.015]),
            offset_frame("j1_child", bodies["lower_arm"], [0.0, 0.0, -0.125]),
            [0.0, 1.0, 0.0],
            damping=0.0,
        )
    )
    # J2: revolute about y at top of lower arm
    plant.AddJoint(
        RevoluteJoint(
            "J2",
            offset_frame("j2_parent", bodies["lower_arm"], [0.0, 0.0, 0.125]),
            offset_frame("j2_child", bodies["upper_arm"], [0.0, 0.0, -0.10]),
            [0.0, 1.0, 0.0],
            damping=0.0,
        )
    )
    # head welded near top of upper arm
    plant.WeldFrames(
        offset_frame("head_mount", bodies["upper_arm"], [0.03, 0.0, 0.11]),
        bodies["head"].body_frame(),
    )
    plant.Finalize()
    return plant, bodies


TRUTH_PLANT, TRUTH_BODIES = build_lamp_plant(TRUE_RHO)
TRUTH_CTX = TRUTH_PLANT.CreateDefaultContext()
KIN_PLANT, KIN_BODIES = build_lamp_plant(np.ones(P))
KIN_CTX = KIN_PLANT.CreateDefaultContext()


# ---------------------------------------------------------------------------
# S2: estimator-side regressor  (kinematics only, no density knowledge)
# ---------------------------------------------------------------------------
def part_centroids_in_S(theta):
    """Centroid of each part in the sensor frame at joint configuration theta."""
    KIN_PLANT.SetPositions(KIN_CTX, np.asarray(theta))
    return np.stack(
        [
            KIN_PLANT.EvalBodyPoseInWorld(KIN_CTX, KIN_BODIES[name]).translation()
            for name, _, _ in PARTS
        ]
    )


def regressor(theta, g_dirs=G_DIRS):
    """Stacked A(theta) in R^{6J x P}: columns g[V_i g_hat; V_i c_i x g_hat]."""
    c = part_centroids_in_S(theta)                       # (P, 3)
    blocks = []
    for g_hat in g_dirs:
        # 예측식도 measure() 와 **같은 부호 규약**을 써야 한다. 한쪽만 뒤집으면
        # y 와 A*rho 의 부호가 어긋나 밀도가 음수로 나온다.
        force_rows = FORCE_SIGN * G_ACC * np.outer(g_hat, VOLUMES)    # (3, P)
        torque_rows = FORCE_SIGN * G_ACC * (np.cross(c, g_hat).T * VOLUMES)
        blocks.append(np.vstack([force_rows, torque_rows]))
    return np.vstack(blocks)


# ---------------------------------------------------------------------------
# Ground-truth measurement: Drake computes total mass and CoM of the true
# plant; the static wrench about the sensor origin follows exactly.
# ---------------------------------------------------------------------------
def measure(theta, g_dirs=G_DIRS, rng=RNG):
    TRUTH_PLANT.SetPositions(TRUTH_CTX, np.asarray(theta))
    m = TRUTH_PLANT.CalcTotalMass(TRUTH_CTX)
    p_com = TRUTH_PLANT.CalcCenterOfMassPositionInWorld(TRUTH_CTX)
    ys = []
    for g_hat in g_dirs:
        f = FORCE_SIGN * m * G_ACC * g_hat
        tau = np.cross(p_com, f)
        noise = rng.normal(0.0, np.sqrt(R_EPS_DIAG))
        ys.append(np.concatenate([f, tau]) + noise)
    return np.concatenate(ys)


# ---------------------------------------------------------------------------
# S4: information gain, S6: update + constrained MAP
# ---------------------------------------------------------------------------
R_STACK_DIAG = np.tile(R_EPS_DIAG, len(G_DIRS))
W_HALF = 1.0 / np.sqrt(R_STACK_DIAG)                     # noise whitener


def info_gain(A, Sigma):
    Aw = A * W_HALF[:, None]
    M = np.eye(A.shape[0]) + Aw @ Sigma @ Aw.T
    return 0.5 * np.linalg.slogdet(M)[1]


def posterior_covariance(Sigma_prev, A):
    Aw = A * W_HALF[:, None]
    info = np.linalg.inv(Sigma_prev) + Aw.T @ Aw
    return np.linalg.inv(info)


def constrained_map(A_all, y_all):
    """Box-constrained MAP over all accumulated data + Gaussian prior."""
    w = 1.0 / np.sqrt(np.tile(R_EPS_DIAG, A_all.shape[0] // 6))
    A_stack = np.vstack([A_all * w[:, None], np.linalg.cholesky(np.linalg.inv(SIGMA0)).T])
    b_stack = np.concatenate([y_all * w, np.linalg.cholesky(np.linalg.inv(SIGMA0)).T @ MU0])
    if HAVE_SCIPY:
        res = lsq_linear(A_stack, b_stack, bounds=RHO_BOUNDS, method="bvls")
        return res.x
    sol, *_ = np.linalg.lstsq(A_stack, b_stack, rcond=None)
    return np.clip(sol, *RHO_BOUNDS)


# ---------------------------------------------------------------------------
# S3: candidate grid (feasibility hooks reduce to joint limits in simulation)
# ---------------------------------------------------------------------------
def candidate_grid(n_per_joint=7):
    axes = [np.linspace(lo, hi, n_per_joint) for lo, hi in JOINT_LIMITS]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
        -1, len(JOINT_LIMITS)
    )
    return [th for th in grid if is_feasible(th)]


def is_feasible(theta):
    # Hook for self-collision / robot IK / payload checks on real hardware.
    return True


def structural_identifiability(candidates):
    C = np.vstack([regressor(th) for th in candidates])
    svals = np.linalg.svd(C * W_HALF_ALL(C), compute_uv=False)
    return svals


def W_HALF_ALL(C):
    return np.tile(W_HALF, C.shape[0] // len(W_HALF))[:, None]


# ---------------------------------------------------------------------------
# Closed loop (S3 -> S6), with a random-selection baseline for comparison
# ---------------------------------------------------------------------------
def run_loop(select, n_rounds, seed):
    rng = np.random.default_rng(seed)
    candidates = candidate_grid()
    Sigma = SIGMA0.copy()
    A_all = np.empty((0, P))
    y_all = np.empty(0)
    history = []
    for t in range(n_rounds):
        theta = select(candidates, Sigma, rng)
        A = regressor(theta)
        y = measure(theta, rng=rng)
        A_all, y_all = np.vstack([A_all, A]), np.concatenate([y_all, y])
        Sigma = posterior_covariance(Sigma, A)
        rho_hat = constrained_map(A_all, y_all)
        history.append(
            dict(
                theta=np.asarray(theta),
                rho=rho_hat,
                rmse=float(np.sqrt(np.mean((rho_hat - TRUE_RHO) ** 2))),
                max_rel_err=float(np.max(np.abs(rho_hat - TRUE_RHO) / TRUE_RHO)),
                sigma_max=float(np.sqrt(np.linalg.eigvalsh(Sigma).max())),
            )
        )
    return history


def select_active(candidates, Sigma, rng):
    gains = [info_gain(regressor(th), Sigma) for th in candidates]
    return candidates[int(np.argmax(gains))]


def select_random(candidates, Sigma, rng):
    return candidates[rng.integers(len(candidates))]


def main():
    print(f"parts: {[p[0] for p in PARTS]}")
    print(f"true densities [kg/m^3]: {TRUE_RHO}")
    print(f"true masses      [kg]  : {np.round(TRUE_RHO * VOLUMES, 4)}\n")

    svals = structural_identifiability(candidate_grid())
    print("structural identifiability (singular values of stacked, whitened C):")
    print(f"  {np.round(svals, 3)}  -> rank {np.sum(svals > 1e-6 * svals[0])}/{P}\n")

    n_rounds, n_random_seeds = 6, 10
    hist_a = run_loop(select_active, n_rounds, seed=1)
    hists_r = [run_loop(select_random, n_rounds, seed=s) for s in range(n_random_seeds)]
    rmse_r = np.array([[h["rmse"] for h in hist] for hist in hists_r])  # (seeds, rounds)

    print(f"{'round':>5} | {'active: theta*':>16} {'RMSE':>8} {'maxRel%':>8} "
          f"| {'random RMSE (mean +/- std over ' + str(n_random_seeds) + ' seeds)':>40}")
    for t, ha in enumerate(hist_a):
        th = np.round(ha["theta"], 2)
        print(f"{t+1:>5} | {str(th):>16} {ha['rmse']:>8.1f} {100*ha['max_rel_err']:>7.1f}% "
              f"| {rmse_r[:, t].mean():>18.1f} +/- {rmse_r[:, t].std():>6.1f}")

    rho_final = hist_a[-1]["rho"]
    print("\nfinal estimate (active selection):")
    for (name, _, rho_true), rho_est in zip(PARTS, rho_final):
        print(f"  {name:<14} true {rho_true:7.1f}  est {rho_est:7.1f}  "
              f"err {100*abs(rho_est-rho_true)/rho_true:5.2f}%")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rounds = np.arange(1, n_rounds + 1)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(rounds, [h["rmse"] for h in hist_a], "o-", label="active (D-optimal)")
        ax.semilogy(rounds, rmse_r.mean(axis=0), "s--", label=f"random theta (mean of {n_random_seeds})")
        ax.fill_between(rounds, rmse_r.mean(0) - rmse_r.std(0),
                        rmse_r.mean(0) + rmse_r.std(0), alpha=0.15, color="C1")
        ax.set_xlabel("measurement round")
        ax.set_ylabel("density RMSE [kg/m$^3$]")
        ax.set_title("Part-wise density identification in Drake")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig("figures/convergence.png", dpi=150)
        print("\nsaved plot -> figures/convergence.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
