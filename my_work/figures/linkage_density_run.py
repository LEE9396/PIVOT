"""2-link / 3-link 를 힌지 고려 / 미고려 두 가지로 돌리고 결과를 JSON 으로 남긴다.

무엇을 비교하나
---------------
실물에는 힌지(명가철물 토크힌지, 실측 41 g)가 **실제로 붙어 있다.** 저울에
올리면 그 무게까지 같이 읽힌다. 달라지는 것은 **추정기가 그걸 아느냐** 뿐이다.

  ignored  추정기 모형에 힌지가 없다. 부위 밀도만 푼다.
           그런데 저울 총무게에는 힌지가 들어 있으므로, 그 몫이 갈 곳이
           없어 부위 밀도로 흘러든다 -> 치우침.
  modelled 힌지를 부위 하나로 같이 푼다 (지금 사양). 저울로 이미 재서
           사전분포가 좁으므로 미지수가 사실상 안 늘어난다.

두 경우 모두 **진리 plant 에는 힌지 질량이 들어 있다.** 그래야 공정하다.

실행:
  ../robot_learning/scripts/run_drake_env.sh python figures/linkage_density_run.py
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import json
from dataclasses import replace

import numpy as np

import density_id_drake as alg
import density_id_objects as obj
import design_core as dc
import study_hinge as sh

HERE = _pathlib.Path(__file__).resolve().parent
ROUNDS, SEEDS, TARGET = 6, 12, 0.01


def part_masses(spec, rho):
    """부위 질량 [kg]. rho 앞쪽 len(parts) 개만 부위다."""
    return np.array([r * p.volume_m3 for p, r in zip(spec.parts, rho)])


def run_case(key, modelled, seed):
    """한 번 돌린다. modelled=False 면 추정기가 힌지를 모른다."""
    full = obj.OBJECTS[key]
    hinge_kg = full.joints[0].hinge_mass_kg

    # 저울이 읽는 값은 언제나 힌지까지 포함한 총질량이다.
    total_true = obj.assembled_mass_kg(full)

    if modelled:
        spec = full
        rho_gt = obj.bind_object(spec)                    # [부위..., 힌지...]
        obj.apply_weight_prior(spec, total_true)
    else:
        # 추정기 쪽 사양에서만 힌지를 뗀다.
        spec = replace(full, joints=[replace(j, hinge_mass_kg=0.0)
                                     for j in full.joints])
        rho_gt = obj.bind_object(spec)                    # [부위...] 뿐
        obj.apply_weight_prior(spec, total_true)
        # 진리 plant 에는 힌지를 점질량으로 달아 둔다 (실물에는 있으니까).
        truth, _ = sh.build_plant_with_hinges(spec, rho_gt, hinge_kg)
        alg.TRUTH_PLANT = truth
        alg.TRUTH_CTX = truth.CreateDefaultContext()

    result = dc.closed_loop(spec, target=TARGET, max_rounds=ROUNDS, seed=seed,
                            n_wanted=len(spec.parts))
    return spec, rho_gt, result


def summarise(key):
    full = obj.OBJECTS[key]
    n_part = len(full.parts)
    gt_rho = np.array([p.rho_gt for p in full.parts])
    gt_mass = part_masses(full, gt_rho)
    out = dict(object=key, label=full.label,
               hinge_g=1000.0 * full.joints[0].hinge_mass_kg,
               n_joint=len(full.joints),
               total_g=1000.0 * obj.assembled_mass_kg(full),
               parts=[p.name for p in full.parts],
               volume_cm3=[p.volume_cm3 for p in full.parts],
               gt_rho=gt_rho.tolist(), gt_mass=gt_mass.tolist())

    for modelled in (False, True):
        name = "modelled" if modelled else "ignored"
        rhos, masses, rounds, worst = [], [], [], []
        hinge_rho = []
        for s in range(SEEDS):
            spec, rho_gt, res = run_case(key, modelled, seed=s)
            rho = res["rho_hat"]
            rhos.append(rho[:n_part])
            masses.append(part_masses(full, rho[:n_part]))
            rounds.append(res["rounds"])
            worst.append(res["worst"])
            if modelled:
                hinge_rho.append(rho[n_part:])
        rho_mean = np.mean(rhos, axis=0)
        mass_mean = np.mean(masses, axis=0)
        out[name] = dict(
            rho=rho_mean.tolist(),
            mass=mass_mean.tolist(),
            rel_mass_err=((mass_mean - gt_mass) / gt_mass).tolist(),
            abs_mass_err_pct=(100 * np.abs(mass_mean - gt_mass) / gt_mass).tolist(),
            rounds=float(np.mean(rounds)),
            claimed_half_pct=100 * float(np.mean(worst)),
            hinge_rho=(np.mean(hinge_rho, axis=0).tolist() if modelled else None),
        )
        print(f"  {key:>6} {name:<9} 라운드 {np.mean(rounds):.1f}  "
              f"부위 질량오차 "
              + ", ".join(f"{v:.2f}%" for v in out[name]["abs_mass_err_pct"])
              + f"   주장반폭 {out[name]['claimed_half_pct']:.2f}%")
    return out


def main():
    obj.set_measurement_averaging()
    data = {}
    print(f"라운드 최대 {ROUNDS}, seed {SEEDS}개 평균, 목표 반폭 {100*TARGET:.0f}%\n")
    for key in ("2link", "3link"):
        data[key] = summarise(key)
        print()
    path = HERE / "linkage_density_data.json"
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    print(f"저장 -> {path}")


if __name__ == "__main__":
    main()
