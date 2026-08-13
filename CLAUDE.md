# Claude Code 안내

이 저장소의 지침은 [AGENTS.md](AGENTS.md) 에 있습니다. **먼저 그 파일을 읽으세요.**

요약만 옮기면:

- 환경 구성은 `./setup/bootstrap.sh` 한 줄.
- 모든 실행은 `my_work/` 에서 `../robot_learning/scripts/run_drake_env.sh python ...`.
  맨 파이썬으로 부르면 pydrake 가 조용히 깨집니다.
- Drake 는 1.54.0 고정. 올리지 마세요.
- GT(정답)는 채점에만. 탐색·정지 판단에 쓰면 안 됩니다.
- 주석은 한국어로, "왜"를 적습니다.
