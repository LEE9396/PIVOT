#!/bin/bash
# p=5 가 끝나면 p=6 을 이어서 돌린다. 세션이 끝나도 살아남는다.
cd /home/junhyeoklee/Desktop/PIVOT/my_work

echo "[$(date +%H:%M:%S)] p=5 종료 대기 중..."
while pgrep -f "study_scaling.py --parts 5" > /dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] p=5 종료 확인. p=6 시작"

../robot_learning/scripts/run_drake_env.sh python -u study_scaling.py \
    --parts 6 --rel 0.05 --seeds 2 --max-rounds 60 --starts 6 \
    --json figures/nlink_v2_p6.json --plot figures/nlink_v2_p6.png \
    > /tmp/nv2_p6.log 2>&1
echo "[$(date +%H:%M:%S)] p=6 종료 (exit $?)"
