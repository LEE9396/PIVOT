# Yuseong-Cheon/PIVOT 에 적용할 프롬프트

아래 내용을 Yuseong-Cheon/PIVOT 체크아웃에서 Claude Code 에 그대로 붙여넣으세요.

---

## 방법 A — 브랜치를 그대로 가져오기 (권장, 5분)

같은 커밋(`2dd7611`) 위에서 만든 변경이라 충돌 없이 붙습니다.

```
LEE9396/PIVOT 의 torque-only-given-mass 브랜치에 있는 커밋 하나를 이 저장소에
가져와줘.

  git remote add lee https://github.com/LEE9396/PIVOT.git
  git fetch lee torque-only-given-mass
  git log --oneline lee/torque-only-given-mass -1     # 171bdb2 인지 확인
  git diff --stat HEAD lee/torque-only-given-mass     # 6개 파일만 바뀌어야 함
  git cherry-pick 171bdb2

부모가 2dd7611 (지금 HEAD) 이라 충돌이 없어야 해. 충돌이 나면 멈추고 어떤
파일에서 났는지 알려줘 — 그건 이 저장소가 그 뒤로 더 나갔다는 뜻이니까.

가져온 뒤 TORQUE_ONLY.md 를 읽고, 아래 두 가지를 확인해줘.

  1) 힘 블록의 계수가 정말 1인지 (문서에 있는 검증 코드)
  2) my_work 에서 hardware_real.self_check() 가 13/13 통과하는지
```

---

## 방법 B — 변경 내용을 직접 적용 (브랜치를 못 가져올 때)

```
이 저장소(PIVOT)의 밀도 추정 파이프라인을 "저울 총질량을 받고 토크 3축만
쓰는" 방식으로 바꿔줘. 파일은 my_work 아래 5개만 건드리면 돼.

## 왜

density_id_drake.regressor 의 힘 행은

    force_rows = FORCE_SIGN * G_ACC * np.outer(g_hat, VOLUMES)

인데, np.outer(g_hat, VOLUMES) 의 세 행이 전부 VOLUMES 의 상수배야. 그래서
힘 블록의 계수(rank)가 1이고, 중력 방향을 늘려도 1 그대로야. 즉 힘 채널이
밀도에 대해 알려주는 건 정확히 한 개, 총질량 Σ V_i ρ_i 뿐이야.

그 하나를 저울에서 받으면 힘 행은 더 줄 게 없어. 반면 실물 힘 채널에는
session_20260904_1736 에서 자세의존 오프셋 58.7 N 이 실렸어 — 램프 전체
무게 5.60 N 의 10 배야. 그런데 SIGMA_F = 0.10 N 로 백색화하니까 그 채널을
실제보다 500 배 믿고 있었어.

## 바꿀 것 — my_work/density_id_drake.py

1. R_EPS_DIAG 정의 바로 뒤에 토크 전용 스위치를 추가해줘.

   - 전역 USE_FORCE_ROWS = True, ROWS_PER_DIR = 6
   - 전역 TOTAL_MASS_KG = None  (저울 값 기록용)
   - noise_diag(sigma_f=None, sigma_t=None, use_force=None)
       USE_FORCE_ROWS 면 [f²]*3 + [t²]*3, 아니면 [t²]*3 을 반환
   - rebuild_noise(sigma_f=None, sigma_t=None)
       SIGMA_F/SIGMA_T 를 갱신하고 R_EPS_DIAG / R_STACK_DIAG / W_HALF 를
       **셋 다** 다시 만든다
   - set_torque_only(enabled=True)
       USE_FORCE_ROWS / ROWS_PER_DIR 을 정하고 rebuild_noise() 를 부른다

   ★ 중요: R_EPS_DIAG / R_STACK_DIAG / W_HALF 는 반드시 **함께** 만들어야 해.
   하나만 갱신하면 백색화 가중치와 회귀행렬의 행 수가 어긋나서 조용히
   틀린 답이 나와.

2. regressor() 에서 USE_FORCE_ROWS 가 False 면 torque_rows 만 쌓게.
3. measure() 도 같게 (np.concatenate([f, tau]) 대신 tau 만).
4. constrained_map() 의 `A_all.shape[0] // 6` 을 `// ROWS_PER_DIR` 로.

## my_work/density_id_objects.py

set_sensor_averaging 안에서 alg.R_EPS_DIAG / R_STACK_DIAG / W_HALF 를 직접
만드는 세 줄을 alg.rebuild_noise(_BASE_SIGMA_F * scale, _BASE_SIGMA_T * scale)
한 줄로 바꿔줘. 여기서 [f]*3+[t]*3 을 다시 쓰면 토크 전용이 6축으로
되돌아가서 sensor_cov 가 (18,18), 야코비안이 (9,9) 로 어긋나 터져.

## my_work/design_core.py

1. torque_rows_only(value, n_dir) 헬퍼를 추가하고 measurement_equation 의
   vector() 안에서 써줘. 6*n_dir 로 들어온 렌치를 3*n_dir 로 자르는 거야.
   ★ 측정·전송·기록은 6축 그대로 두고 **여기서만** 자를 것. 그래야 원시
   렌치가 보존되고 |F| ~ M g 검산도 계속 돼.

2. grasp_columns 에서 alg.USE_FORCE_ROWS 가 False 면 np.zeros((3,3)) 힘
   블록을 빼줘.

3. grasp_map 에 grasp_mu_m=None 인자를 추가해줘.
   - 사전분포 목표를 np.zeros(3) 대신 grasp_mu / grasp_sigma_m 으로
   - 상자도 grasp_mu ± grasp_bound_m 으로 (지금은 0 둘레)

4. tls_map 에도 같은 grasp_mu_m 인자를 추가해줘.
   - grasp0 기본값을 grasp_mu 로
   - 상자를 (grasp_mu ± 0.05) 로
   - 잔차 항을 (grasp - grasp_mu[:n_grasp]) / grasp_sigma_m 로

   ★ 왜: GRASP_SIGMA_M 이 5 mm 인데 실제 파지점 어긋남이 173.9 mm 였어.
   35 시그마라서 MAP 이 정답을 강하게 벌주고, 상자 한계 ±50 mm 에
   [50, -50, 50] 으로 세 축 전부 붙었어 (세션 기록의 부호까지 일치).
   grasp_columns 자체는 처음부터 옳았고 중심과 폭만 틀렸던 거야.

## my_work/dual_view.py

1. 인자 3개 추가:
   --total-mass-kg FLOAT     저울로 잰 총질량 [kg], 힌지 포함
   --torque-only / --use-force   (dest=torque_only, 기본 None)
   --grasp-mu-mm X Y Z       FoundationPose 가 잰 파지점 어긋남 [mm]

2. parse_args() 직후, 다른 어떤 코드보다 먼저:
   - torque_only 가 None 이면 (args.hardware == "real") 로 정한다
   - alg.set_torque_only(args.torque_only) 를 부른다
   - --hardware real 인데 --total-mass-kg 가 없으면 parser.error 로 막는다
     (없으면 자산 GT 로 총질량을 만들게 되어 "정답 넣고 정답 맞히기"가 됨)
   - grasp_mu_m = args.grasp_mu_mm * 1e-3

   ★ 반드시 파싱 직후여야 해. 아래서 regressor() 가 한 번이라도 불리면
   행 수가 굳어지고 그 뒤에 바꾸면 백색화와 어긋나.

3. prepare() 에 total_mass_kg=None 인자를 추가하고, prior == "weight" 갈래에서
   그 값이 있으면 obj.assembled_mass_kg(spec, rho_gt) 대신 그것을 쓰게.
   alg.TOTAL_MASS_KG 에도 넣어줘. 어느 쪽을 썼는지 화면에 찍고.

4. PlannerScreen.__init__ 에 total_mass_kg=None, grasp_mu_m=None 추가:
   - self.grasp_mu_m 을 두고 self.grasp_hat 의 초기값으로
   - self.fixed_mass_kg 를 두고, self.total_mass_kg 를 그 값으로 고정

5. 라운드 갱신부에서 self.total_mass_kg = float(alg.VOLUMES @ self.rho_hat)
   를 fixed_mass_kg 가 None 일 때만 하게.

   ★ 왜: 라운드마다 다시 계산하면 (V @ rho_hat) 의 추정 오차가 파지점 열의
   계수로 되먹임돼서 두 미지수가 서로를 흉내내. 토크만 쓰면 총질량이 유일한
   규모 기준이라 특히 그래.

6. grasp_map / tls_map 호출에 grasp_mu_m=self.grasp_mu_m 을 넘겨줘.

7. read_one() 의 시뮬레이션 파지점 오차 주입에서 wrench[3:6] 을
   alg.USE_FORCE_ROWS 에 따라 [3:6] 또는 [0:3] 이 되게 해줘.

8. 예보 루프에서 alg.R_EPS_DIAG 를 직접 만드는 자리를 alg.noise_diag(...) 로,
   finally 의 복원 뒤에 alg.rebuild_noise() 를 한 번 불러줘.

## my_work/pivot_ui.py

1. "--prior", "water" 하드코딩을 "weight" 로 바꿔줘.
   water 는 등방 사전분포라 총질량을 전혀 안 묶어. 그래서 부위 밀도가
   하한 50 에 붙어도 못 막았고, 총질량이 416.9 g vs 저울 571.0 g 로 27%
   부족하게 나왔어.

2. 설정 파일에서 넘길 수 있게:
   ("--total-mass-kg", "TOTAL_MASS_KG") 를 옵션 루프에 추가
   GRASP_MU_MM 이 있으면 --grasp-mu-mm 으로 (공백 구분 3개)
   USE_FORCE 가 참이면 --use-force

## 검증 (반드시 다 돌리고 결과를 보여줘)

1. 힘 블록의 계수가 1인지:

   import numpy as np, density_id_drake as alg
   A6 = alg.regressor(np.array([0.4, -0.7]))
   F = np.vstack([A6[6*i:6*i+3] for i in range(3)])
   sv = np.linalg.svd(F, compute_uv=False)
   print("rank", int(np.sum(sv > 1e-9*sv[0])))          # 1 이어야 함
   _,_,Vt = np.linalg.svd(F)
   u = alg.VOLUMES/np.linalg.norm(alg.VOLUMES)
   print("|cos|", abs(Vt[0] @ u))                        # 1.0 이어야 함

2. 토크 전용 전환 후 모양 일관성:

   alg.set_torque_only(True)
   A3, y3 = alg.regressor(th), alg.measure(th)
   assert A3.shape[0] == y3.size == alg.W_HALF.size == 3*len(alg.G_DIRS)
   # 그리고 A3 이 A6 의 토크 부분과 정확히 같아야 함

3. hardware_real.self_check() 가 13/13 통과

4. CLI 가드: --hardware real 인데 --total-mass-kg 없으면 거부되는지

5. 복원 성능 — 파지점을 173.9 mm 어긋뜨리고 grasp_map 으로 되찾기.
   기대값 (5부위 시뮬 물체, 재분배 사전분포 = 평균밀도):

     6축,   명목 0     최대오차 205%   총질량오차 18.8%   railing [-11, 50, -50]
     토크만, 명목 0     최대오차 343%   총질량오차 26.1%   railing
     6축   + 명목값     최대오차  56%   총질량오차 0.03%   [-54.1, 82.0, -147.2]
     토크만 + 명목값     최대오차  57%   총질량오차 0.00%   [-53.8, 82.1, -146.7]

   ★ 남은 56% 는 사전분포 수축이야 (재분배 폭을 x10 하면 양쪽 다 4~6% 로
   떨어져). 추정기 결함이 아니니 여기서 더 파지 마.

## 하지 말 것

- 하드웨어·타어·전송 경로에서 렌치를 6축에서 3축으로 줄이지 마. 원시 렌치가
  보존돼야 오프라인 재검증이 되고 |F| ~ M g 검산도 돼.
- SIGMA_F / SIGMA_T 값 자체는 건드리지 마.
- 아직 안 풀린 문제(각도 3도, 손-눈 오차, 메시 프레임 규약)를 여기서 같이
  고치려 하지 마. 별도 작업이야.
```

---

## 적용 후 실물 실행

```bash
python dual_view.py --mode deploy --hardware real \
    --object desklamp --grasp pinch --grasp-part link_3 \
    --grasp-frame measured \
    --total-mass-kg <저울값 kg> \
    --grasp-mu-mm <FoundationPose X Y Z, mm> \
    --grasp-sigma-mm 15 \
    --prior weight
```

`pivot_ui` 설정 파일:

```
TOTAL_MASS_KG=0.571
GRASP_MU_MM=-49.2 82.0 -145.2
GRASP_SIGMA_MM=15
```

`--grasp-sigma-mm` 을 10 이 아니라 **15** 로 둔 이유: FoundationPose 위치
오차 10 mm 에 손-눈 캘리브레이션 오차(위치 2~5 mm, 자세 0.5~1° → 팔 뻗은
거리 0.5 m 에서 8.7 mm)가 더해집니다. 실제 값은 재보고 정하십시오.
