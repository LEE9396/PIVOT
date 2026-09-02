# 팀원 PC 체크리스트

실물 실험까지 해야 할 것 전부. **순서대로** 하고, 각 항목의 통과 기준을 확인한
뒤 다음으로 간다. 앞이 막히면 뒤는 의미가 없다.

`$R` = `../robot_learning/scripts/run_drake_env.sh`
★ = 이걸 빠뜨리면 **에러 없이 조용히 틀린 결과**가 나온다

---

## 0. 정리 — 어느 트리를 쓸 것인가

지금 PC 에 네 개가 있다. 어느 것이 진짜인지 먼저 정해야 한다.

```
~/Desktop/PIVOT              assets/final_objects/lamp 를 가짐
~/Desktop/PIVOT-real-setup   런처가 실제로 쓰던 것 (traceback 이 가리킴)
~/MeshPCA                    clean
~/MeshPCA-real-setup         두 conf 가 모두 이쪽을 가리킴
```

- [ ] **PIVOT 을 하나로 합친다.** 두 체크아웃이 갈라진 채로 둔 것이 이번
      사고의 근본이었다 — 한쪽에만 pinch 보정이 있었다.
- [ ] 미커밋 작업을 먼저 백업한다: `git stash` 또는 `git diff > ~/backup.patch`
- [ ] `git status` 로 지워지면 안 되는 것이 있는지 확인

## 1. 코드 받기

- [ ] PIVOT
      ```bash
      cd ~/Desktop/PIVOT
      git fetch origin && git checkout real-experiment-ready
      ```
- [ ] ★ **MeshPCA 쪽 두 파일도 반드시 적용한다.** PIVOT 저장소의
      `integration/` 에 들어 있다. 이걸 빠뜨리면 창 2 가 여전히 죽고,
      타어에 직접교시 모드가 없다.
      ```bash
      cp ~/Desktop/PIVOT/integration/foundationpose/make_desk_lamp_masks.py \
         ~/MeshPCA-real-setup/foundationpose/
      # tare_real.py 는 아래 5번을 먼저 읽을 것
      ```
- [ ] `$R python setup/doctor.py` 전부 통과

## 2. 램프 자산

- [ ] 자산 폴더를 이 배치로 만든다 (`assets/final_objects/lamp/README.md` 참고)
      ```
      lamp/standlamp*.urdf
      lamp/visual_meshes/{base,support,head}.obj
      lamp/collision_meshes/{base,support,head}.obj
      lamp/collision_meshes/convex/<부위>/part_NN.obj
      ```
- [ ] ★ **볼록 조각이 메시와 같은 빌드인지 확인한다.**
      저장소에 넣어 둔 15개 조각은 **2026-09-01 빌드**용이다.
      그 빌드의 메시를 쓴다면 그대로 쓰고, 다른 빌드면 다시 구워라.
      ```bash
      python -m pip install coacd
      $R python tools/make_convex.py <lamp>/collision_meshes --report
      ```
      "가장 긴 조각이 부위의 절반을 넘는다" 경고가 나오면
      `--threshold` 를 절반으로 낮춰 다시.

      > 두 빌드는 관절 원점이 **50~90 mm** 다르다. 섞으면 조용히 틀린다.

- [ ] ★ `setup/experiment.conf` 에서 **`LAMP_ASSET_DIR` 과 `FP_MESH_DIR` 이
      같은 트리**를 가리키게 한다. 하나만 바꾸면 충돌 기하와 창 2 오버레이가
      다른 좌표계가 된다.

## 3. 장비 없이 검증 — 여기까지 로봇 없이 다 된다

- [ ] ★ **그쪽 빌드로 다시 확인한다.** 이쪽 검증은 08-25 빌드로 했다.
      ```bash
      $R python tools/verify_objects.py --json results/verify.json
      ```
      통과 기준: 세 물체 모두 도달 각도 1개 이상 + "사슬 연결됨"
      참고값(08-25 빌드): 2link 2/3 · 3link 8/9 · 램프 6/9, 최소 간격 11.00 mm
      → 램프가 0/9 면 `long_axis` 나 자산 배치를 의심한다
- [ ] `$R python tools/preflight.py --conf setup/experiment.conf`
      캘리브레이션 항목은 4번 뒤에 통과한다
- [ ] `$R python my_work/pivot_ui.py --dry-run`
- [ ] `$R python my_work/hardware_real.py --check` (10/10)
- [ ] `$R python my_work/gripper_hw.py --check` (9/9)
- [ ] `$R python my_work/grasp_overlay.py --check` (7/7)

## 4. 캘리브레이션 — 로봇 정지

### 4a. 핸드아이 (DGIST EasyHeC, Robotiq 자산판)

- [ ] `third_party/robotiq_arg85_description` 이 PIVOT 에 있는지 확인.
      없으면 `https://github.com/a-price/robotiq_arg85_description` 를 받는다
- [ ] `prepare_robotiq_assets.py --pivot-root <PIVOT>` 로 자산 생성
      (기본 경로가 그쪽 PC 절대경로다. 반드시 명시)
- [ ] `run_robotiq.sh` 의 `EASYHEC_PYTHON` 을 conda python 으로 수정
- [ ] `config_robotiq.yaml` 의 IP(`192.168.50.51`)·카메라 serial 확인
- [ ] 실물 치수 확인: AFT200 52.3 mm / ⌀80 mm, Robotiq 개구 78.5 mm
- [ ] 캡처 10~20장 → `annotate_masks` → `init_pose --search` → `solve` →
      `render_check` (mean IoU 0.9 이상)
- [ ] 변환
      ```bash
      python integration/MeshPCA/import_easyhec.py \
          --transform <EasyHeC>/models/.../Tc_c2b.txt \
          --intrinsics <EasyHeC>/data/.../K.txt \
          --output ~/MeshPCA/calibration/handeye.json
      $R python my_work/import_calibration.py --input ~/MeshPCA/calibration/handeye.json
      ```
- [ ] 검산: `Tc_c2b @ X_WC` 가 단위행렬

> **카메라를 옮겼다면 이전 결과를 재사용하지 마라.** 그 저장소 문서에도
> 예전 ChArUco 값이 "카메라 이동으로 무효(렌더 0 px)" 가 된 기록이 있다.

### 4b. 책상 평면 (카메라만, 로봇 명령 0개)

- [ ] ```bash
      $R python my_work/calibrate_table_rgbd.py \
          --handeye calibration/<핸드아이>.json \
          --output calibration/rb5_table_current.json
      ```
- [ ] 통과 신호: 실행 화면에 `테이블 실측 반영: 윗면 z ... mm`
- [ ] ★ `[주의] 테이블이 명목값입니다` 가 뜨면 아직 **도면값(750 mm)** 으로
      도는 중이다. 명목이 실제보다 낮으면 부딪히는 자세를 통과시키고,
      그 접촉력이 F/T 에 잡혀 **밀도로 흡수된다**

### 4c. 창 2 마스크

- [ ] `./setup/launch_experiment.sh` 로 창 2 를 띄운다
- [ ] 부위 이름표가 자동으로 구워지고 수동 박스 모드로 뜬다
      (평판 = base, 굽은 팔 = support, 라이트바 = head)
- [ ] 통과 기준: `/tmp/lamp_foundationpose_live/latest.json` 생성

### 4d. 각도 부호·영점

- [ ] 관절마다 한 번씩, 40초간 끝에서 끝까지 천천히 접는다
      ```bash
      $R python tools/angle_signs.py --model-only --joint 1     # 먼저 곡선 확인
      $R python tools/angle_signs.py --pose-file <latest.json> \
          --joint 1 --seconds 40 --output calibration/angle_signs.json
      ```
- [ ] `--pose-key` 를 물어보면 `latest.json` 의 실제 키 이름을 지정
- [ ] ★ **나온 잔차를 `experiment.conf` 의 `ANGLE_FLOOR_DEG` 에 넣는다.**
      기본 0.5° 는 실측(1~3°)보다 낙관적이라 정보이득이 나쁜 각도를 고른다

## 5. 타어 — 세팅 때 한 번

- [ ] ★ **먼저 확인: 그쪽 `tare_real.py` 가 어느 버전인가.**
      `aft_tare_final_20260901_2219.json` 은 `joint_deg`·`repeat_delta` 를
      담고 있는데, essential.zip 의 어느 `tare_real.py` 도 그 키를 안 쓴다.
      **세 번째 버전이 있고 그게 실제로 쓰이는 코드다.**
      ```bash
      grep -l joint_deg $(find ~ -name tare_real.py 2>/dev/null)
      ```
      찾으면 `integration/meshpca/tare_real.py` 의 `--manual` / `--verify`
      부분만 떼어 거기에 붙인다. 통째로 덮어쓰지 마라.
- [ ] ★ **기존 타어는 구성이 맞는지 확인.** Robotiq + AFT200 로 잰 것이
      맞다면(=현재 구성) 재사용 가능. XHand 로 잰 것이면 무효다
- [ ] 세팅 때 한 번 (직접교시 — 로봇에 명령하지 않는다)
      ```bash
      PIVOT_WORKDIR=$PWD $R python <tare_real.py> --manual --setup \
          --output calibration/aft_tare_current.json --overwrite
      ```
- [ ] 실험 직전마다 10초
      ```bash
      PIVOT_WORKDIR=$PWD $R python <tare_real.py> --verify \
          --output calibration/aft_tare_current.json
      ```
      통과: `힘 드리프트 < 0.10 N` / 초과: 다시 `--setup`

- [ ] **미해결 하나 확인해 달라.** 기존 타어 파일의 관절각으로 FK 를 돌려
      실제 중력방향을 복원하면 라벨과 **정확히 90° 어긋난다**(3개 중 2개).
      새로 `--setup` 하면 `direction_error_deg` 가 기록되므로 그때 판별된다.
      값이 1° 안이면 정상, 90° 근처면 프레임 규약이 어긋난 것이다

## 6. 실험

- [ ] 리허설: `./setup/launch_experiment.sh --rehearse`
- [ ] 책상 확인 사격: 모형상 TCP 가 책상 위 50 mm 인 자세에서 천천히
      내려 접촉 지점 확인. 35 mm 에 닿으면 책상이 15 mm 높은 것
- [ ] 본 실험
      ```bash
      $R python my_work/pivot_ui.py --conf setup/experiment.conf
      # 또는 기존 4창
      ./setup/launch_experiment.sh
      ```
- [ ] 물체 순서: **2-link → 3-link → 램프**.
      앞의 둘은 정답을 아니까 채점이 되고, 거기서 방법이 맞는 것을 보인 뒤
      램프로 간다

---

## 막히면

| 증상 | 먼저 볼 것 |
| --- | --- |
| "힌지·도달·충돌을 모두 통과하는 자세가 없다" | `tools/verify_objects.py` — 어느 쌍이 몇 mm 겹치는지 나온다 |
| 창 2 가 안 뜨고 3분 뒤 종료 | `/tmp/pivot_win2.log`. 마스크 실패면 `MANUAL_MASK=1` |
| 라운드가 중간에 날아감 | 이제 시작 자세를 경유한다. 그래도 나면 경로 로그 |
| 결과가 그럴듯한데 이상함 | `preflight` 의 **주의** 항목부터. 명목값으로 도는 중이다 |

**preflight 에서 주의가 뜨면 "일단 돌려보자"가 아니라 "결과를 믿을 수 없다"로
읽어라.** 이 실험에서 위험한 것은 프로그램이 멈추는 게 아니라 에러 없이 틀린
답이 나오는 것이다.

---

## 이쪽에서 아직 확인 못 한 것

| | 내용 |
| --- | --- |
| 🔶 | `pivot_ui` 1·3단계 **화면** — Meshcat 뷰와 로봇 씬은 그쪽에서 처음 뜬다 |
| 🔶 | `export_urdf.py` 인자 규약 — 5단계에서 실패하면 여기 |
| 🔶 | `tare_real --manual/--verify` 의 **장비 통신부** (`rbpodo.CobotData`) |
| 💭 | `ANGLE_TOL_RAD = 1.0°` — IK 가 경계에 붙어 푼다. 타어·실험 각각 1° → 렌치의 4 %. 조일 값어치 있음 |
| 💭 | 타어 모형 잔차 4.35 N (램프 무게의 78 %). 케이블 당김 의심. 자세별 표를 쓰는 한 상쇄되므로 지금은 무해 |
