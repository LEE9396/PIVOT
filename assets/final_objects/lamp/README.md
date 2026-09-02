# 스탠드 조명 자산 배치

`desk_lamp.py` 의 `LAYOUT == "final"` 이 읽는 배치다.

```
lamp/
  standlamp*.urdf                       배달물의 URDF
  visual_meshes/{base,support,head}.obj  화면용 (FoundationPose 도 이걸 쓴다)
  collision_meshes/{base,support,head}.obj  통짜 물리 메시
  collision_meshes/convex/<부위>/part_NN.obj  **볼록 조각** ← 여기 들어있는 것
```

## 큰 메시는 저장소에 없다

`visual_meshes/`, `collision_meshes/*.obj` 는 합쳐서 54 MB 라 넣지 않았다.
배달물(`Lamp_Final.zip` 또는 `.../standlamp/`)에서 복사해라.

## 볼록 조각은 여기 있다 — 그리고 어느 빌드인지가 중요하다

들어있는 15개 조각은 **2026-09-01 빌드**(`standlamp_2dgs_profile_c_angle_limited`)
용이다. 좌표가 맞는지 관절점으로 검증했다.

```
09-01 빌드 URDF 와 대조   관절점 → 재료  4.7 ~ 7.9 mm    ✅ 같은 빌드
08-25 빌드 URDF 와 대조   관절점 → 재료 66.5 ~ 100.4 mm  ❌ 다른 빌드
```

**다른 빌드의 메시와 섞으면 안 된다.** 두 빌드는 관절 원점이 50~90 mm 다르다.
새 빌드를 받으면 조각도 새로 구워라:

```bash
python -m pip install coacd
$R python tools/make_convex.py <lamp>/collision_meshes --report
```

"가장 긴 조각이 부위의 절반을 넘는다" 경고가 나오면 `--threshold` 를 절반으로
낮춰 다시 구워라.

## 왜 조각이 필요한가

Drake 의 `Convex` 는 파일 하나를 **볼록 껍질 하나로** 감싼다. 통짜 메시를 넘기면
형상이 1.6~2.6 배 부푼다. 2026-09-02 에 이것이 조용히 일어나 실물 실험이 하루
날아갔다. 지금은 조각이 없으면 `desk_lamp.collision_meshes()` 가 **예외를 던진다.**
