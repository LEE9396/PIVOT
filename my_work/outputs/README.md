# outputs — 파이프라인이 만들어 내는 것

여기 있는 파일은 **다시 만들 수 있습니다.** 손으로 고치지 마세요.

| 파일 | 만드는 것 |
| --- | --- |
| `estimated_desklamp.urdf` | `export_urdf.py --object desklamp` |
| `estimated_3link.urdf` | `export_urdf.py --object 3link` |
| `plan_3link.json`, `plan_2link.json` | `robot_scene.py --plan outputs/plan_3link.json` |
| `results_objects.txt`, `results_mg_plastic_hinge.txt` | `density_id_objects.py` 검증 기록 |

## 램프 URDF 는 배달물을 덮어쓰지 않습니다

`estimated_desklamp.urdf` 는 **배달물 원본**
(`assets/desk_lamp_minimal_sim/drake/object.urdf`) 을 그대로 복사한 뒤
링크별 `<inertial>` 셋만 추정값으로 바꾼 것입니다. 형상·관절·볼록분해는
원본 그대로이고, 메시 경로만 이 폴더 기준 상대경로로 다시 쓰여 있습니다.

배달물 폴더 자체는 건드리지 않습니다 (AGENTS.md).
