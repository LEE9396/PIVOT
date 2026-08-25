#!/bin/bash
# 지금 도는 seed 4 실행이 끝나면, seed 8 로 최종 표를 만든다.
cd /home/junhyeoklee/Desktop/PIVOT/my_work
echo "[$(date +%H:%M:%S)] seed 4 실행 종료 대기..."
while pgrep -f "study_scaling.py --parts 2 3 4 5 6 --rel 0.05 --seeds 4" > /dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] seed 8 최종 실행 시작"
../robot_learning/scripts/run_drake_env.sh python -u study_scaling.py \
    --parts 2 3 4 5 6 --rel 0.05 --seeds 8 --max-rounds 30 --starts 6 --target 0.02 \
    --json figures/nlink_final.json --plot figures/nlink_final.png \
    > /tmp/nfinal.log 2>&1
echo "[$(date +%H:%M:%S)] seed 8 종료 (exit $?)"
