# 로봇 PC — 환경 구성부터 통합 UI 실행까지 한 번에 시키는 프롬프트

로봇(RB5)·AFT200·D456 이 연결된 PC 의 Yuseong-Cheon/PIVOT 체크아웃(`~/Desktop/PIVOT`)에서
Claude Code 를 열고 아래를 붙여넣는다. **첫 줄의 무게만 바꾼다.**

```
램프 총질량(저울, 힌지 포함) = 0.569 kg   # 팀 실측 569 g (tools/LOCAL_FT_VALIDATION.md). 오늘 저울 값으로 바꿔도 됨

이 저장소(PIVOT, 로봇 PC)에서 데스크 램프 부위 밀도 실험을 통합 UI 로 바로 시작할 수
있게 환경을 만들고 UI 를 띄워줘. 순서대로 하고, 각 단계의 출력을 짧게 보여준 뒤 다음으로.
막히면 어느 단계에서 왜 막혔는지와 마지막 30줄을 보여주고 멈춰. 아래 "하지 말 것"은 절대 어기지 마.

1) 코드
   git log --oneline -1 로 HEAD 를 보여줘. LEE9396/PIVOT 의 torque-only-given-mass 브랜치를
   가져와 cherry-pick 해 (작업 트리가 더러우면 먼저 알려줘):
     git remote add lee https://github.com/LEE9396/PIVOT.git 2>/dev/null || true
     git fetch lee torque-only-given-mass
     git cherry-pick $(git merge-base HEAD lee/torque-only-given-mass)..lee/torque-only-given-mass
   충돌이 나면 멈추고 파일명을 알려줘. 끝나면 grep -n '"--prior", "weight"' my_work/pivot_ui.py 가
   잡히는지 확인해.

2) 검증 (Drake 환경은 robot_learning/scripts/run_drake_env.sh 로만)
     robot_learning/scripts/run_drake_env.sh python -m unittest discover -s tools -p 'test_*.py'
     robot_learning/scripts/run_drake_env.sh python tools/round_check.py --self-test
   32 OK 와 self-test 판정 OK/FAIL 두 줄이 나와야 해.

3) 설정 파일
   setup/experiment.conf 가 있으면 그대로 두고(이 PC 의 경로·IP 가 들어 있다), 없으면
   setup/experiment.conf.example 을 복사한 뒤 PIVOT_ROOT / MESHPCA_ROOT / FOUNDATIONPOSE_PYTHON /
   ROBOT_HOST / AFT_HOST / GRIPPER_PORT / LAMP_ASSET_DIR / FP_MESH_DIR 를 나에게 물어서 채워.
   키 수정은 setup/quickstart_real.sh 가 하니 직접 고치지 마.

4) 장애물 파일
   calibration/workspace_obstacles_current.json 이 없으면 calibration/workspace_obstacles_example.json
   을 보여주고, 받침대·단차 같은 추가 장애물의 위치·크기(로봇 베이스 기준, m)를 나에게 물어서
   만들어. 내가 "없다"고 하면 boxes 를 [] 로, status 를 validated 로 써.

5) 준비 점검 (로봇은 아직 안 움직임)
     ./setup/quickstart_real.sh --mass 0.569 --check
   preflight 표에서 실패 행이 있으면 그 행의 "->" 처방을 그대로 보여주고 멈춰.
   타어가 없거나 0.5 N 게이트에 떨어지면 --check 에서는 아직 안 재니, 그 사실만 알려줘.

6) 실행 (여기서 손목이 움직일 수 있다 — 타어가 필요하면 5초 카운트다운 뒤 자동으로 잰다)
   내가 "그리퍼 비었고 케이블 고정했다, 시작해" 라고 말하기 전에는 이 명령을 실행하지 마.
   말하면:
     ./setup/quickstart_real.sh --mass 0.569
   를 백그라운드로 실행하고 로그 경로와 대시보드 주소(http://localhost:8080)를 알려줘.
   타어 결과(tare_check 힘/토크 잔차, 센서 축)와 런처 점검 결과를 로그에서 뽑아 보여줘.

7) UI 진행 중 내가 보는 것
   - 1 파지점 단계 뒤: 로그의 "회전 차이 N°" 를 보여줘. 30° 넘어 스스로 멈췄으면
     LAMP_ASSET_DIR 와 FP_MESH_DIR 가 같은 트리인지 확인해 줘.
   - 4 탐색 뒤: 로그의 "라운드 검산" 표(힘 크기 / 잔차팽창 / 파지 오프셋 / 원시 렌치)를 보여줘.
     하나라도 실패면 그 행의 "->" 를 보여주고, 어떤 원인인지 SESSION_20260904_ROOT_CAUSE.md 기준으로
     한 줄로 말해줘.

8) 끝나면
   sessions/<세션>/experiment_results.json 의 converged, 라운드 수, 부위 질량(link_3 / link_1 / link_2)
   을 GT 85 / 399 / 87 g 와 나란히 표로 보여줘. converged 가 false 면 "센서·기준점 문제"인지
   "도심 오차(연구 문제)"인지 라운드 검산 결과로 구분해서 말해줘.

하지 말 것
 - 다른 PC 의 calibration/*.json 이나 타어 파일을 복사하지 마.
 - setup/wrist_tare.sh --run-force 를 쓰지 마 (검산 불합격 타어로 실험 금지).
 - experiment.conf 의 GRASP_MU_MM 에 값을 넣지 마 (measured 프레임에서 이중 계산).
 - dual_view 에 --auto-adjust 를 넘기지 마 (실물에서는 사람이 램프를 접는다).
 - 6) 의 실행 명령은 내가 시작하라고 말하기 전에 돌리지 마.
```

## 이 프롬프트가 끝나면 남는 형님 일

파지(램프를 그리퍼에 물리기) · 각도(추천값으로 접기) · 경로 승인 — 전부 UI 가 시킬 때.
그리고 처음 한 번, AFT200 렌치 기준면이 `ft_mount` 원통 중심과 같은지 데이터시트 확인.
