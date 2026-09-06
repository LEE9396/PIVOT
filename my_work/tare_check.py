"""영점 조정(3자세 tare)이 물리적으로 말이 되는지 검산한다.

왜 이게 필요한가
----------------
영점 조정은 "물체를 안 잡은 상태로 중력 3방향을 밟으며 그리퍼 무게를 재두는"
절차다. 지금까지는 재서 저장하고 끝이었다. 맞는지 확인을 안 했다.

그런데 **그리퍼 무게는 이미 아는 값이다** (Robotiq 2F-85 = 0.925 kg, 커플러
포함 약 1.0 kg). 즉 영점 조정은 그 자체로 "알려진 추를 세 방향에서 재는" 검증
실험이다. 추를 따로 달 필요가 없다. 세 줄을 아래 두 모형에 맞춰보기만 하면
된다.

    힘   :  f = b + W * g_hat            미지수 4개(b 3, W 1), 식 9개
    토크 :  tau = c + (m*r) x g          미지수 6개(c 3, m*r 3), 식 9개

여기서 g_hat 은 **중력 방향을 센서 좌표계로 표현한 것**이다 (robot_scene 의
IK 가 거는 조건과 같은 규약). b, c 는 자세와 무관한 고정 바이어스다.

이 맞춤 하나로 네 가지가 동시에 판정된다.

    W 의 부호      센서가 중력 방향(+g)을 읽나, 떠받치는 방향(-g)을 읽나.
                   추정기 모형(density_id_drake.measure)은 +g 를 가정한다.
                   실측이 -g 면 밀도가 음수로 밀려 하한에 붙는다.
    |W| / 9.81     힘의 크기와 단위. 그리퍼 질량과 맞아야 한다.
    |r|            토크의 기준점. 그리퍼 무게중심까지의 거리이므로 3~8 cm 여야
                   한다. 미터 단위로 나오면 토크가 파지점이 아닌 다른 원점
                   기준으로 들어오고 있다는 뜻이다.
    맞춤 잔차      영점 자체의 품질. 강체 하나가 중력 방향만 바꾼 것으로
                   설명되지 않는 몫이며, 케이블 장력·마운트 응력·온도 드리프트가
                   여기로 나온다. 이 값이 물체 무게에 육박하면 영점을 빼는
                   행위가 오차를 **줄이는 게 아니라 주입한다.**

실행
----
    python tare_check.py <tare.json>            # 판정만
    python tare_check.py <tare.json> --strict   # 불합격이면 종료코드 1
"""

import json
import sys

import numpy as np

def model_force_sign():
    """추정기가 가정하는 힘 부호. 못 읽으면 저장소 기본값 +1.0.

    density_id_drake 는 pydrake 를 끌고 오므로, 이 검산을 드레이크 환경 밖에서도
    돌릴 수 있게 실패를 허용한다.
    """
    try:
        import density_id_drake as alg
        return float(alg.FORCE_SIGN)
    except Exception:                                       # noqa: BLE001
        return +1.0


# Robotiq 2F-85 (0.925 kg) + AFT 커플러. 실물이 다르면 --tool-kg 로 준다.
DEFAULT_TOOL_KG = 1.0
TOOL_KG_TOL = 0.25              # 카탈로그값 대비 허용 폭
G_ACC = 9.81

# 합격 문턱.
#
# 힘 0.5 N 은 "재려는 물체(램프 571 g = 5.6 N)의 10 % 이하" 에서 왔다. 이보다
# 크면 영점 오차가 신호를 가린다. 같은 자세 반복측정 편차가 0.034 N 이므로
# 0.5 N 은 센서 성능이 아니라 배선·마운트 상태를 보는 값이다.
FORCE_RESIDUAL_MAX_N = 0.5
TORQUE_RESIDUAL_MAX_NM = 0.02
LEVER_MIN_M, LEVER_MAX_M = 0.02, 0.10
# 센서 축이 모형과 이보다 더 돌아가 있으면 장착 변환이 틀린 것이다.
AXIS_ANGLE_MAX_DEG = 5.0


def _skew(v):
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def fit_force(g_dirs, forces):
    """f = b + W*g 를 최소제곱으로 맞춘다. (바이어스, W, 잔차노름)"""
    rows, rhs = [], []
    for g, f in zip(g_dirs, forces):
        block = np.zeros((3, 4))
        block[:, :3] = np.eye(3)
        block[:, 3] = g
        rows.append(block)
        rhs.append(f)
    A, y = np.vstack(rows), np.concatenate(rhs)
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    return x[:3], float(x[3]), float(np.linalg.norm(y - A @ x))


def fit_torque(g_dirs, torques):
    """tau = c + (m*r) x g 를 맞춘다. (바이어스, m*r, 잔차노름)

    (m*r) x g = -[g]x (m*r) 이므로 미지수에 대해 선형이다.
    """
    rows, rhs = [], []
    for g, t in zip(g_dirs, torques):
        block = np.zeros((3, 6))
        block[:, :3] = np.eye(3)
        block[:, 3:] = -_skew(np.asarray(g, float)) * G_ACC
        rows.append(block)
        rhs.append(t)
    A, y = np.vstack(rows), np.concatenate(rhs)
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    return x[:3], x[3:], float(np.linalg.norm(y - A @ x))


def solve_sensor_axes(g_dirs, forces):
    """센서 축이 모형과 얼마나 어긋나 있는지 푼다. (R, 공구무게, 잔차)

    모형은 "중력이 센서 좌표계에서 g_hat 이다" 라고 믿는다. 실제 센서 축이
    거기서 R 만큼 돌아가 있으면 읽히는 힘은

        f_k = b + W * (R @ g_hat_k)

    가 된다. 방향끼리 빼면 고정 바이어스 b 가 사라지므로

        f_i - f_j = W * R * (g_hat_i - g_hat_j)

    이고, 이건 직교 프로크루스테스 문제다 (SVD 한 번). 방향이 3개면 겨우
    맞아떨어져 검산할 여유가 없고, 5개 이상이면 남는 잔차로 "이 답을 믿어도
    되나" 까지 판정된다.

    R 이 항등에 가까우면 축이 맞는 것이고, 크게 돌아가 있으면 robot_scene 의
    GRIPPER_MOUNT_YAW_RAD / MakeXRotation 이 실물과 다른 것이다.
    """
    g_dirs = np.asarray(g_dirs, dtype=float)
    forces = np.asarray(forces, dtype=float)
    A, B = [], []
    for i in range(len(g_dirs)):
        for j in range(i + 1, len(g_dirs)):
            A.append(g_dirs[i] - g_dirs[j])
            B.append(forces[i] - forces[j])
    A, B = np.asarray(A).T, np.asarray(B).T          # (3, 쌍수)

    def procrustes(target):
        U, sigma, Vt = np.linalg.svd(target @ A.T)
        # 반사(det=-1)는 회전이 아니다. 마지막 특이값 부호를 뒤집어 막는다.
        D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
        R = U @ D @ Vt
        s = float(np.sum(sigma * np.diag(D)) / max(np.sum(A * A), 1e-12))
        return R, s, float(np.linalg.norm(target - s * R @ A))

    # 공구 무게항의 부호(+g 냐 -g 냐)는 회전으로 흡수되지 않는다. 순수한
    # 부호 뒤집기는 det=-1 이라 회전이 아니기 때문이다. 그래서 양쪽을 다
    # 풀어 보고 잘 맞는 쪽을 고른다. 고른 부호가 곧 센서의 힘 규약이다.
    R_pos, s_pos, res_pos = procrustes(B)
    R_neg, s_neg, res_neg = procrustes(-B)
    if res_neg < res_pos:
        return R_neg, -s_neg, res_neg
    return R_pos, s_pos, res_pos


def rotation_angle_deg(R):
    """회전 행렬이 몇 도짜리 회전인가."""
    return float(np.degrees(np.arccos(
        np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def load(path):
    data = json.loads(open(path, encoding="utf-8").read())
    entries = data["entries"]
    g = np.array([e["g_hat"] for e in entries], dtype=float)
    w = np.array([e["wrench"] for e in entries], dtype=float)
    return data, g, w


AXIS_GAIN_RATIO_MAX = 1.10        # 축별 이득이 이보다 더 벌어지면 축척 문제다


def solve_sensor_map(g_dirs, forces):
    """제약 없이 f = b + M g 를 맞추고, M 이 무엇인지 진단한다.

    왜 회전부터 맞추면 안 되나
    -------------------------
    solve_sensor_axes 는 M 이 "무게 * 회전" 이라고 **가정하고** 가장 가까운
    회전을 찾는다. 그 가정이 깨진 자료를 넣으면 깨졌다고 말하지 않고, 설명이
    안 되는 어긋남을 회전각으로 둔갑시킨다.

    실제로 그랬다. 축 이득이 6.5 배 어긋난 자료에 "179.4 deg 회전" 이라는
    답이 나왔고, 그대로 믿고 좌표계를 돌렸으면 더 나빠졌을 것이다.

    그래서 아무 제약 없이 먼저 맞춘다. M 은 9 개 수이고 방향이 4 개 이상이면
    풀린다. 그러고 나서 M 을 뜯어본다.

        축별 이득    M 의 각 행 크기. 셋이 같아야 한다 (회전은 크기를 보존).
        행렬식       +1 이면 회전, -1 이면 거울상 = 어느 한 축의 부호가 뒤집혔다.
        자유 잔차    이것이 작으면 **측정 자체는 깨끗하다.** 크면 케이블 등
                     자세에 따라 변하는 외력이 있는 것이다. 둘은 전혀 다른
                     문제인데 회전만 맞춰서는 구별이 안 된다.
    """
    g_dirs = np.asarray(g_dirs, dtype=float)
    forces = np.asarray(forces, dtype=float)
    design = np.hstack([np.ones((len(g_dirs), 1)), g_dirs])
    solution, *_ = np.linalg.lstsq(design, forces, rcond=None)
    bias, matrix = solution[0], solution[1:].T
    residual = forces - design @ solution
    free_res = float(np.sqrt((residual ** 2).sum() / residual.size))

    gains = np.linalg.norm(matrix, axis=1)          # 센서 축별 이득 [N]
    ratio = float(gains.max() / max(gains.min(), 1e-12))
    unit = matrix / np.maximum(gains, 1e-12)[:, None]
    det = float(np.linalg.det(unit))

    # 거울상이면 어느 한 축의 부호만 뒤집어도 회전이 된다. 다만 **세 축
    # 중 어느 것인지는 힘 자료만으로 못 가린다** — 셋 다 똑같이 잘 맞는다.
    # 하나를 골라 주면 고른 티가 안 나므로, 셋을 다 적고 사람이 정하게 한다.
    # 나오는 회전각이 모형이 이미 쓰는 값(예: 90 deg)에 가까운 쪽이 유력하다.
    flips = []
    if det < 0:
        for axis in range(3):
            signs = np.ones(3)
            signs[axis] = -1.0
            candidate = signs[:, None] * unit
            u, _, vt = np.linalg.svd(candidate)
            proper = u @ np.diag([1.0, 1.0,
                                  float(np.sign(np.linalg.det(u @ vt)))]) @ vt
            flips.append(dict(axis="xyz"[axis],
                              angle_deg=rotation_angle_deg(proper),
                              gap=float(np.linalg.norm(candidate - proper))))

    return dict(bias=bias, matrix=matrix, free_residual_n=free_res,
                gains_n=gains, gain_ratio=ratio, det=det, flips=flips,
                is_rotation=(det > 0 and ratio <= AXIS_GAIN_RATIO_MAX))


def check(path, tool_kg=DEFAULT_TOOL_KG, log=print):
    """영점 파일 하나를 검산한다. (합격여부, 항목별 결과) 를 돌려준다."""
    data, g_dirs, wrench = load(path)
    forces, torques = wrench[:, :3], wrench[:, 3:]

    bias_f, w_signed, res_f = fit_force(g_dirs, forces)
    bias_t, m_r, res_t = fit_torque(g_dirs, torques)

    mass = abs(w_signed) / G_ACC
    sign = "-g (떠받치는 힘)" if w_signed < 0 else "+g (중력 방향)"
    model_sign = model_force_sign()
    lever = float(np.linalg.norm(m_r) / mass) if mass > 1e-6 else np.inf

    rows = [
        ("힘 부호",
         np.sign(w_signed) == model_sign,
         f"센서 {sign}   모형 FORCE_SIGN = {model_sign:+.0f}",
         f"어긋났다. density_id_drake.FORCE_SIGN 을 {-model_sign:+.1f} 로 "
         "바꾸세요. **그 한 줄만** 바꿉니다 — 읽는 쪽에서 또 뒤집으면 "
         "원위치가 되고, 시뮬레이션 결과는 이 값과 무관하게 같습니다"),
        ("공구 질량",
         abs(mass - tool_kg) <= TOOL_KG_TOL,
         f"{mass:.3f} kg  (기대 {tool_kg:.2f} +/- {TOOL_KG_TOL:.2f})",
         "안 맞으면 힘의 크기나 단위가 틀렸다"),
        ("토크 지렛대",
         LEVER_MIN_M <= lever <= LEVER_MAX_M,
         f"{1000*lever:.1f} mm  (기대 {1000*LEVER_MIN_M:.0f}~{1000*LEVER_MAX_M:.0f})",
         "미터 단위로 나오면 토크가 파지점이 아닌 원점 기준이다"),
        ("힘 맞춤 잔차",
         res_f <= FORCE_RESIDUAL_MAX_N,
         f"{res_f:.3f} N  (허용 {FORCE_RESIDUAL_MAX_N})",
         "케이블 장력·마운트 응력·온도 드리프트. 케이블을 팔에 고정하고 다시 재라"),
        ("토크 맞춤 잔차",
         res_t <= TORQUE_RESIDUAL_MAX_NM,
         f"{res_t:.4f} N·m  (허용 {TORQUE_RESIDUAL_MAX_NM})",
         "위와 같음"),
    ]

    # 방향이 4개 이상이면 센서 축까지 풀어 본다.
    axes, mapping = None, None
    if len(g_dirs) >= 4:
        mapping = solve_sensor_map(g_dirs, forces)
        rows.append((
            "축별 이득",
            mapping["gain_ratio"] <= AXIS_GAIN_RATIO_MAX,
            f"x {mapping['gains_n'][0]:.2f}  y {mapping['gains_n'][1]:.2f}"
            f"  z {mapping['gains_n'][2]:.2f} N"
            f"  (최대/최소 {mapping['gain_ratio']:.2f})",
            "회전은 크기를 보존하므로 셋이 같아야 한다. 벌어졌으면 좌표계가 "
            "아니라 **센서 값을 N 으로 바꾸는 축척**이 틀린 것이다. "
            "센서 드라이버의 축별 배율을 확인하세요"))
        rows.append((
            "좌표계 방향",
            mapping["det"] > 0,
            f"행렬식 {mapping['det']:+.3f}"
            + ("" if mapping["det"] > 0 else
               "  (한 축 부호를 뒤집으면 회전이 됩니다: "
               + ", ".join(f"{f['axis']} -> {f['angle_deg']:.1f} deg"
                           for f in mapping["flips"]) + ")"),
            "음수면 거울상이다. 회전으로는 절대 못 만든다 — 어느 한 축의 "
            "부호가 뒤집혀 있다는 뜻이다. 세 축 중 어느 것인지는 힘 자료만으로 "
            "못 가리니, 나온 회전각이 모형이 쓰는 값에 가까운 쪽을 택하고 "
            "실물 배선으로 확인하세요"))

        # 회전 가정이 성립할 때만 회전각을 말한다. 안 그러면 설명 안 되는
        # 어긋남이 회전각으로 둔갑해 엉뚱한 곳을 고치게 된다.
        if mapping["is_rotation"]:
            R, signed, res_axis = solve_sensor_axes(g_dirs, forces)
            axes = dict(R=R, force_n=signed, residual_n=res_axis,
                        angle_deg=rotation_angle_deg(R))
            rows.append((
                "센서 축 정렬",
                axes["angle_deg"] <= AXIS_ANGLE_MAX_DEG,
                f"모형 대비 {axes['angle_deg']:.1f} deg 회전"
                f"  (잔차 {res_axis:.3f} N)",
                "robot_scene 의 GRIPPER_MOUNT_YAW_RAD / MakeXRotation 이 실물과 "
                "다릅니다. 여기서 나온 회전으로 바꾸세요"))

    log(f"[영점 검산] {path}")
    if len(g_dirs) < 4:
        log(f"  [주의] 방향이 {len(g_dirs)}개뿐입니다. 미지수와 식의 수가 거의"
            " 같아 잔차가 실제보다 작게 나오고,\n"
            "         센서 축은 풀 수 없습니다. --setup 을 5방향 이상으로"
            " 돌리세요.")
    log(f"  고정 바이어스  힘 {np.round(bias_f, 2)} N"
        f"   토크 {np.round(bias_t, 3)} N·m")
    if mapping is not None:
        # 이 숫자가 작으면 '측정은 깨끗한데 해석이 틀린 것'이고, 크면
        # '재는 동안 무언가가 건드린 것'이다. 완전히 다른 문제다.
        log(f"  제약 없이 f = b + M g 를 맞춘 잔차"
            f" {mapping['free_residual_n']:.3f} N"
            + ("   -> 측정 자체는 깨끗합니다. 해석(축척·부호·좌표계)이 문제입니다."
               if mapping["free_residual_n"] <= FORCE_RESIDUAL_MAX_N else
               "   -> 재는 동안 자세에 따라 변하는 외력이 있습니다 (케이블 등)."))
    for name, ok, value, hint in rows:
        mark = "통과" if ok else "실패"
        log(f"  [{mark}] {name:<12} {value}")
        if not ok:
            log(f"         -> {hint}")

    # 파일이 좌표 회전을 선언해 놓고 코드가 안 쓰면 조용히 틀린다.
    if data.get("sensor_to_pivot") is not None:
        rot = np.array(data["sensor_to_pivot"], dtype=float)
        if not np.allclose(rot, np.eye(3)):
            log("  [주의] 파일에 sensor_to_pivot 회전이 들어 있는데 코드가 읽지"
                " 않습니다. 저장된 wrench 가 이미 회전된 값인지 확인하세요.")

    if axes is not None and axes["angle_deg"] > AXIS_ANGLE_MAX_DEG \
            and res_f > FORCE_RESIDUAL_MAX_N:
        log(f"  [참고] 위의 '힘 맞춤 잔차' {res_f:.2f} N 은 축이 맞다고 보고 잰"
            " 값이라 축 회전 몫이 섞여 있습니다.\n"
            f"         축까지 풀고 남은 잔차는 {axes['residual_n']:.3f} N"
            " 입니다. 축을 먼저 고치세요.")

    passed = all(ok for _, ok, _, _ in rows)
    log(f"  => {'합격' if passed else '불합격 — 이 상태로 측정하면 안 됩니다'}")
    return passed, dict(bias_force=bias_f, bias_torque=bias_t,
                        tool_mass_kg=mass, force_sign=np.sign(w_signed),
                        lever_m=lever, residual_force_n=res_f,
                        residual_torque_nm=res_t, axes=axes, rows=rows,
                        n_directions=len(g_dirs))


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    tool = DEFAULT_TOOL_KG
    if "--tool-kg" in argv:
        tool = float(argv[argv.index("--tool-kg") + 1])
    ok, _ = check(argv[0], tool_kg=tool)
    return 0 if ok or "--strict" not in argv else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
