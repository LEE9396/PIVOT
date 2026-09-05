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


def load(path):
    data = json.loads(open(path, encoding="utf-8").read())
    entries = data["entries"]
    g = np.array([e["g_hat"] for e in entries], dtype=float)
    w = np.array([e["wrench"] for e in entries], dtype=float)
    return data, g, w


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

    log(f"[영점 검산] {path}")
    log(f"  고정 바이어스  힘 {np.round(bias_f, 2)} N"
        f"   토크 {np.round(bias_t, 3)} N·m")
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

    passed = all(ok for _, ok, _, _ in rows)
    log(f"  => {'합격' if passed else '불합격 — 이 상태로 측정하면 안 됩니다'}")
    return passed, dict(bias_force=bias_f, bias_torque=bias_t,
                        tool_mass_kg=mass, force_sign=np.sign(w_signed),
                        lever_m=lever, residual_force_n=res_f,
                        residual_torque_nm=res_t, rows=rows)


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
