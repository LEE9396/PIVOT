# 빈 공구/물체 파라미터 차감 — 2026-09-07

Scalable Real2Sim의 **빈 상태 식별 → 하중 상태 식별 → 파라미터 차감**을
PIVOT의 정지 손목 F/T 측정에 적용했다. 논문의 KUKA 관절 토크 기반 동적 식별을
그대로 이식한 것은 아니다. 정지 측정으로 식별하는 값은 질량 m, 1차 모멘트
h=m·COM, 빈 상태 센서 바이어스 b다. 회전 관성은 식별하지 않는다.

근거: [논문 §III-D/IV-4](https://arxiv.org/html/2503.00370v2#S3.SS4),
[원본 파라미터 차감 코드](https://github.com/nepfaff/robot_payload_id/blob/af163600fb1a552ad6be0c708749225001ab5f1c/scripts/identify_grasped_object_payload.py#L382).

## 계산과 사용 범위

```text
빈 상태:   y_empty(g) = b + A_static(g) [m_tool, h_tool]
하중 상태: y_load(g)  = b + A_static(g) [m_total, h_total]
물체:      m_object = m_total - m_tool
           h_object = h_total - h_tool
           COM_object = h_object / m_object
```

하중 상태의 b는 빈 상태에서 식별한 값으로 고정한다. 물체 힘을 새 영점으로
흡수하지 않는다. 모든 h와 토크는 동일한 ft_mount 원점/축 기준이다.

밀도 추정용 입력은 실제 `y_load - y_empty_model`이다. 맞춤값 `A_static @ params`로
측정값을 대체하지 않는다. 큰 수직 힘 잔차를 감춰 밀도 계산을 통과시키지 않는다.
합격한 결과의 `estimator_reply`는 기존 `design_core.measurement_equation`의
`wrench`, `wrench_raw`, `tare_applied`, `tare_required` 규약을 따른다.
관절 달린 물체는 **같은 파지점과 같은 물체 관절각**을 유지하는 측정끼리만
한 번에 식별한다. 관절각을 바꾸면 loaded configuration-id를 새로 부여한다.

현재는 기록·오프라인 분리 도구이며 통합 UI의 영점을 자동 교체하지 않는다.
`passed`는 모델 식별/잔차 검사만 의미한다. 실물 검증 전이므로
`calibration_valid=false`, `uncertainty_calibrated=false`다. 새 공구 모델의 통계적
공분산을 밀도 불확실성에 전달하는 연결은 아직 없다. 기존 3°/10 mm 설정을
센서·공구 교정의 불확실성으로 대신 사용하지 않는다.

## 수집

1. 실제 장애물/optical table 평면을 확인한 뒤, 충돌 없는 측정 자세를 계획한다.
   self collision, wrist 케이블 원통, 장애물과 전체 이동 경로를 검사해야 한다.
   이 도구는 로봇 이동을 수행하지 않는다.
2. 빈 공구로 충분히 다른 중력 방향에서 기록한다. 최소 3방향이어도 행렬의
   rank/조건수 검사를 통과해야 한다. 같은 자세를 오래 기록해도 방향 수가 늘지 않는다.
3. 같은 센서 영점·장착 상태를 유지하고 같은/가까운 그리퍼 개구로 물체를 잡는다.
   파지점·물체 관절각을 고정하고 대응 자세에서 기록한다.
4. 물체 제거 후 빈 상태 복귀와, 식별에 사용하지 않은 알려진 하중을 별도로
   검증해야 실제 교정 정확도를 평가할 수 있다.

창3의 `hardware_status.json`에서 그리퍼 위치 카운트를 읽는다. Robotiq 위치
카운트를 임의로 mm로 환산하지 않는다. 2초보다 오래된 피드백, 구동 중 상태,
fault 또는 측정 중 1카운트 초과 변화는 거부한다. 시리얼 포트에는 새로 연결하지 않는다.

현재 자세에서 빈 상태를 기록하는 예 (STATUS는 실행 중인 창3의 실제 파일 경로):

```bash
./setup/local_ft_check.sh --record --samples 100 --blocks 10 --settle-s 5 \
  --gripper-status-file "$STATUS" --ft-session-id ft_session_01 \
  --load-state empty --configuration-id empty_opening_150 \
  --label empty_pose_1 --output /home/cheon/Desktop/PIVOT/my_work/outputs/empty_pose_1.json
```

`setup/local_ft_check.sh`는 my_work로 이동하므로 **--output은 절대 경로를 권장**한다.
나머지 자세도
명시적으로 이동/정지 확인 후 기록한다. 하중 기록은 `--load-state loaded
--configuration-id bottle_A`로 바꾼다. 세션 중 영점 초기화·재연결·장착 변경이
있으면 ft-session-id를 새로 부여하고 빈 상태부터 다시 수집한다.

## 분리 실행

my_work에서 (예시 파일 이름은 실제 기록으로 바꿀 것):

```bash
../robot_learning/scripts/run_drake_env.sh python -B ../tools/identify_ft_payload.py \
  --empty outputs/empty_pose_1.json outputs/empty_pose_2.json outputs/empty_pose_3.json \
  --loaded outputs/load_pose_1.json outputs/load_pose_2.json outputs/load_pose_3.json \
  --known-mass-kg 1.332 --output outputs/bottle_A_separated.json
```

빈 공구 configuration-id별로 그룹을 만들고 실제 개구가 가장 가까운 그룹을
선택한다. 기본 허용 개구 차이는 3카운트다. 하중 기록은 하나의 configuration-id만
허용한다. 알려진 질량은 결과의 절대/상대 오차 평가에만 쓰며 식별에는 넣지 않는다.

기본 검사 한계는 각 자세의 힘 잔차 노름 0.5 N, 토크 0.02 N·m,
열 크기로 정규화한 행렬 조건수 100이다. 구간 평균의 최대 변화도 같은 힘/토크
한계로 검사한다. 이 값은 조정 가능한 공학적 검사 기준이며 센서의 공인 정확도나
1σ가 아니다. 불합격 시 종료코드 2, `estimator_reply=null`로 저장한다.
기존 출력 파일을 덮어쓰지 않는다.

## 이번 검증

- 합성 1.332 kg에서 공구/물체 질량·무게중심 복원, 토크 변화 시 질량 유지 확인.
- 하중에 추가 Fz 18.84 N을 넣으면 실패하며 센서 바이어스가 변경되지 않음.
- 같은 자세 반복, 다른 파지/영점 세션, 개구 불일치, 오래된 그리퍼 상태 거부 확인.
- 기존 A/B/A는 새 메타데이터가 없고 한 방향 기록이므로 새 교정으로 승격하지 않음:
  `outputs/payload_identification_legacy_aba_check_20260907.json`.
- 과거 +Y 자세에서 J4/J5/J6 각각 ±2° 후보 7개를 계산한 **정보량 검사**는
  정규화 조건수 115.14로 기본 기준 100을 넘었다. 무잡음 합성값에도 정보량 기준으로
  거부된다. 이는 이동/충돌 검사가 아니며 실제 로봇을 움직이지 않았다:
  `outputs/payload_small_motion_observability_20260907.json`.

따라서 ±2°의 작은 왕복만으로 전 방향 공구 모델을 충분히 식별했다고 주장하지
않는다. 실물에서는 안전한 자세 후보 중 정보량을 확보하거나, 같은 자세 직접
차감으로 검증 범위를 제한해야 한다. 현재 활성 영점 파일은 변경하지 않았다.

## 10분 드리프트 기록과 구동 구간 배제

`local_ft_check.py --record --duration-s 600`은 실제 경과시간으로 수집하며
매 구간 `.partial.json`을 원자적으로 저장한다. `--samples 100 --hz 50`이면
구간은 약 2초다. 원시 6축, 제어기 6축, 수신 시각, 관절각, 그리퍼 상태를 보존한다.
전체 기록 동안 0.1°를 넘는 자세 변화, 개구/명령 변화, 오래되거나 구동 중인
그리퍼 피드백은 실패 처리한다. 센서 영점 초기화 명령은 보내지 않는다.

```bash
../robot_learning/scripts/run_drake_env.sh python -B ../tools/analyze_ft_drift.py \
  outputs/ft_empty_drift_10min_20260907.json \
  --output outputs/ft_empty_drift_10min_20260907.analysis.json \
  --plot outputs/ft_empty_drift_10min_20260907.png
```

첫/마지막 60초 평균 차이, 시간 기울기, 원시 표준편차, 2초/20초 평균 산포를
구분한다. 온도를 측정하지 않았으므로 온도 원인을 확정하지 않는다. 시간 상관이
있는 평균들의 산포를 표본 수의 제곱근으로 다시 줄이거나 밀도 1σ로 쓰지 않는다.

통합 UI는 `dual_view --gripper-status-file .../hardware.json`을 전달한다.
실측 센서 평균은 최신 gSTA=3, gOBJ=1/2/3, fault=0 및 비구동 상태를 확인한 뒤
추가 1초 기다리고, 수집 중 개구/요청 위치가 바뀌면 거부한다. 최신 피드백 주기는
창3의 약 0.5초에 제한된다. 기존 실행 중 프로세스에는 재시작 후 적용된다.
다른 실행 경로에서는 이 옵션을 지정해야 창3 피드백 검사가 활성화된다.

## 무엇이 보정되고 무엇이 남는가

| 항목 | 분리 방법 | 판정의 한계 |
| --- | --- | --- |
| 공구 중력과 토크 | 빈 상태의 m, h, 고정 바이어스 식별 후 현재 실제 g에서 차감 | 개구/장착/영점 변경 시 재식별 |
| 물체 중력 토크 | 제거하지 않고 밀도 행렬의 토크 행에 입력 | 파지점·도심·각도 오차가 토크 잔차에 영향 |
| 시간에 따른 영점 변화 | 정지 장시간 기록과 하중 전후 빈 상태 비교 | 하중 중 영점 재설정 또는 임의 추세 제거 금지 |
| 힘·토크 출력 간 간섭 | 독립된 알려진 힘/모멘트의 교정 및 별도 검증 필요 | 이 정적 공구 차감만으로 제거되지 않음 |

빈 공구의 자세별 출력 변화는 정상 중력 성분일 수도 있다. Fz와 Mx/My가 함께
변했다고 간섭이 입증되는 것도 아니다. **빈 상태 차감 후, 실제 g에 수직인 힘과
모형 토크 잔차**를 함께 봐야 한다. 강체의 무게중심 이동은 토크를 바꾸지만 같은
총질량의 순중력 힘을 늘리지 않는다. 장착 변형에 의한 출력 간섭은 별도 가설이다.

제조사 [AFT200-D80-C 매뉴얼](https://emanual.oopy.io/aft-200-d80-c-eng)
(2026-04-28 개정)은 힘/토크 분해능 0.15 N/0.015 N·m, 별도의 STD 항목,
예열 권고와 온도 보정 유무별 bias/송신 모드 짝을 명시한다. 이 센서에 적용되는
리비전과 RB 제어기의 설정은 아직 확인하지 못했다. CAN 원시 변환식을 이미
환산된 RB Modbus 레지스터에 다시 적용하면 안 된다. 공개 자료에서 정량적
crosstalk 보증치를 확인하지 못했으므로 분해능을 그 값으로 대신하지 않는다.

Scalable Real2Sim의 차감 원칙은 공구와 물체의 **정상 강체 성분**을 분리한다.
합성 PIVOT `measure()`에 공구를 더한 뒤 차감하면 원래 6축 및 밀도 행렬과
일치하며 1.332 kg을 복원한다. 실물의 하중 의존 잔차가 사라진다는 증명은 아니다.
현재 전 방향의 실물 교정/검증 자료가 없으므로 통합 UI의 영점 교체와 전 방향
정확도 합격은 보류 상태다. 별도 하중/자세로 검증 후에만 활성화한다.
