# 2026-09-08 재현 패키지 검증

로봇 이동이나 센서 영점 변경 없이 수행했다.

| 검사 | 결과 |
| --- | --- |
| tools/test_*.py, 현재 작업 트리 | 32 tests OK |
| Git index에 올린 파일만 임시 디렉터리에 추출 후 같은 테스트 | 32 tests OK |
| pivot_ui.py --dry-run, 별도 /tmp 세션 | 준비~내보내기 6단계 전이 OK |
| bootstrap.sh --check | 11/12 통과. 기존 프로세스가 7000 포트를 사용해 포트 항목만 실패 |
| Git index 추출본에서 doctor.py | 같은 11/12. 보정 파일 없이 명목 씬 생성과 자산 검사 통과 |
| MeshPCA 패치 | 깨끗한 4911bef HEAD에 apply --check/apply 성공, 변경·신규 11파일이 원 PC와 byte 단위 일치 |
| SAM3 패치 | 깨끗한 46957e4 HEAD에 apply --check/apply 성공, 변경 1파일이 원 PC와 byte 단위 일치 |
| launch_experiment.sh / wrist_tare.sh / local_ft_check.sh | bash -n 통과 |

임시 checkout 검사는 기존 Drake 가상환경을 DRAKE_VENV_DIR로 지정했다.
새 PC에서 패키지를 인터넷으로 설치하거나 GPU 코드를 재컴파일한 검사는 아니다.
기존 코드의 메시 파일 핸들 ResourceWarning은 발생했으나 테스트 실패는 없었다.
테이블·카메라 명목값으로 씬을 만들 수 있다는 사실은 실물 경로 검증을 대신하지 않는다.

새 PC에서 남은 인수 검증: 독립 Python 환경 설치, CUDA 확장 import 및 실제 추론,
USB/Ethernet 장치 연결, 핸드아이·상판·장애물·타어 검증, UI 승인 절차와 실측 실험.
