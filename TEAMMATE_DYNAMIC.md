# 팀원 PC — 동적 여기 식별 실험 (2026-09-11)

> 이 파일의 "프롬프트" 절을 그대로 AI 어시스턴트에게 붙여 넣어도 되고, 사람이 순서대로 해도 된다.
> 정적 방법(dual_view)은 그대로 남아 있다. 이 문서는 **바뀐 방법**만 다룬다.

무엇이 바뀌었나 (한 문단). 논문의 4단계 탐색이 "정지 3방향 측정"에서 "정지 3방향 + 8초 여기 궤적"으로
바뀌었다. 부위마다 질량·무게중심·관성 10개를 풀되, 관절이 미끄러지지 않도록 힌지 축 토크를
제약해 궤적을 설계한다. 관절마다 4개 조합은 어떤 실험으로도 안 보이며 시뮬레이션에도 영향이
없다 — 자산 파일이 그 방향을 표시한다. 자세한 것은 `my_work/dynex/README.md`.

## 순서 (앞이 안 되면 뒤는 의미가 없다)

```
0. 저장소·환경           git pull, bootstrap, doctor
1. 모의로 끝까지         pivot_ui.py --sim --auto  (장비 없이 4라운드)
2. 센서 100 Hz 확인       Aft200Stream 으로 60 초 정지 기록 → 잡음·분해능·드리프트
3. 툴 식별 ★             물체 없이, 실험 벌림량으로  → calibration/tool_dynamic.json
4. 아는 강체 검사 ★       질량·무게중심을 아는 덩어리 → 질량 2 %, 무게중심 2 mm 안
5. (선택) 힌지 유지토크   토크 게이지가 있으면 HINGE_TORQUE 로. 없으면 자동 모드가 버틴 만큼만 시도한다
6. 3링크 실험            pivot_ui.py (METHOD=dynamic), 라운드 4개, 세션 폴더 통째로 공유
```

★ 표시 둘이 없으면 6번 결과는 믿을 수 없다. 툴/물체 질량비가 2.7 이라 툴 질량 1 % 오차가
물체 2.7 % 로 샌다.

## 프롬프트

````text
PIVOT 저장소에서 "동적 여기 식별" 실험을 이 PC 에서 진행하려고 해. 순서대로 해 주고,
로봇을 실제로 움직이는 명령은 실행 전에 나에게 확인받아. 모든 python 실행은
my_work/ 에서 ../robot_learning/scripts/run_drake_env.sh python ... 으로 한다 (맨 python 금지).
Drake 1.54 고정.

━━ 0. 저장소·환경 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  cd ~/Desktop/PIVOT && git fetch origin && git checkout real-experiment-ready && git pull
  ./setup/bootstrap.sh
  $R python setup/doctor.py                      # 전부 통과
  먼저 읽을 것: TEAMMATE_DYNAMIC.md (이 파일), my_work/dynex/README.md 6·7절,
  my_work/dynex/GIVENS_AND_RISKS.md (실물에서 직접 재야 하는 값).

━━ 1. 모의로 끝까지 (장비 없이) ━━━━━━━━━━━━━━━━━━━━━━━━
  cd my_work
  $R python pivot_ui.py --conf ../setup/experiment_sim.conf --sim --auto
  통과 기준: 세션 폴더에 angle/path/wrench/posterior_round_1..4 와
  export/asset_dynamic.json 이 생기고, 라운드마다 "σ(정적조합)" 이 줄어든다.
  Meshcat 주소가 출력되면 브라우저로 열어 로봇이 움직이는 것을 본다.

━━ 2. 센서 100 Hz 확인 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  $R python - <<'PY'
  import sys, time, numpy as np; sys.path.insert(0, '.')
  import hardware_real as hr
  s = hr.Aft200Stream('192.168.50.51', 100.0); time.sleep(60); d = s.drain()
  t = np.array([x[0] for x in d]); w = np.vstack([x[1] for x in d])
  print('실효 속도 %.1f Hz' % (len(t)/(t[-1]-t[0])), '잡음 STD', w.std(0).round(4), '드리프트', (w[-100:].mean(0)-w[:100].mean(0)).round(4))
  PY
  기대: 100 Hz 근처, 힘 STD ≤ 0.4 N, 토크 STD ≤ 0.025 N·m. 훨씬 크면 케이블·접지·마운트.

━━ 3. 툴 식별 (물체 없이, 실험 벌림량) ━━━━━━━━━━━━━━━━━━━━
  그리퍼를 3링크 파지 벌림량(약 45 mm)으로 닫아 두고, 팔을 열린 공간에 세운다.
  $R python -m dynex.tool_id --hardware real --amp 0.15 --n-traj 3 --opening-m 0.045 \
      --robot-host 192.168.50.51 --aft-host 192.168.50.51 --out calibration/tool_dynamic.json
  첫 궤적은 --amp 0.15 (느림). 잘 따라가면 0.25. 궤적 전에 Enter 를 물으니 비상정지를 들고 한다.
  통과 기준: 잔차 RMS 가 힘 ≤ 0.5 N, 토크 ≤ 0.03 N·m. 크면 케이블 힘·시각 지연·추종 지연.
  주의: 궤적 스트리밍(rb5_stream, move_servo_j 5 ms)은 장비 없이 작성된 미검증 코드다.
  안 움직이거나 튀면 hardware_real.py 끝의 rb5_stream/_servo_j 를 rbpodo 예제
  (examples/move_servo_j.cpp: t1=0.01 t2=0.1 gain=1 alpha=1, 5 ms) 에 맞춰 고친다.

━━ 4. 아는 강체 검사 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  저울로 잰 쇠막대/블록을 같은 벌림량으로 잡고 (무게중심 위치를 자로 잰다):
  $R python -m dynex.tool_id --hardware real --amp 0.15 --baseline calibration/tool_dynamic.json \
      --expect-mass-kg <저울 kg> --expect-com-mm <x y z, 센서 퍽 중심 기준 mm> --out /tmp/payload_check.json
  통과 기준: 질량 2 % 안, 무게중심 2 mm 안. 틀리면 토크 기준점(AFT200 퍽 중심 가정)이나
  시각 지연을 의심 — 이것이 3링크보다 먼저 잡아야 할 오차다.

━━ 5. 힌지 유지토크 (선택) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  토크 게이지가 있으면 두 힌지의 미끄러지기 시작하는 토크를 재서 experiment.conf 에
  HINGE_TORQUE=<N·m> 로 넣는다 (상한으로만 쓰인다). 없으면 비워 둔다: 세션이 "이미 버틴
  최대 토크"를 인증값으로 기억하고 그 1.25 배까지만 다음 자세·궤적을 시도한다. 첫 라운드는
  0.05 N·m 부터라 느리고, 카메라로 각도가 2° 이상 움직인 것이 보이면 그 데이터를 버리고
  상한을 내린다. 그래서 물체가 떨어지거나 관절이 크게 돌아가는 일은 없다.

━━ 6. 3링크 실험 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  setup/experiment.conf: OBJECT=3link GRASP_PART=link0_base METHOD=dynamic SENSOR=datasheet_100hz
    HINGE_TORQUE=<있으면> TOOL_FILE=calibration/tool_dynamic.json TOTAL_MASS_KG=<저울 kg>
    MAX_ROUNDS=4 GRIPPER_FORCE=205 + 기존 ROBOT_HOST/AFT_HOST/FP_OUTPUT
  창 2(FoundationPose)는 3링크 메시로 latest.json 에 joint1/joint2 각도를 내야 한다.
  $R python pivot_ui.py --conf ../setup/experiment.conf
  라운드마다: 추천 각도로 관절을 맞춘다 → 로봇이 정지 3자세를 잰다 → 8초 여기 궤적을
  미리보기로 보여주고 승인 버튼을 기다린다 → 실행 → 궤적 뒤 관절각이 2° 이상 움직였다는
  경고가 나오면 미끄러진 것이니 HINGE_TORQUE 를 낮춰 다시 한다.
  끝나면 세션 폴더(my_work/sessions/session_*) 를 통째로 공유한다. wrench_round_N.csv 가
  있으면 오프라인에서 다시 추정할 수 있다.

━━ 규칙 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - 정답(GT)은 채점에만. 탐색·정지 판단에 쓰지 않는다.
  - 코드를 고치면 무엇을 왜 바꿨는지 세션 폴더의 NOTES.md 에 적는다.
  - 로봇이 처음 궤적을 따라갈 때는 --amp 0.15, 비상정지, 작업 공간 비우기.
````
