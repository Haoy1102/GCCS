
# TaskSchedule (IDE version)

- 直接在 IDE 里运行根目录的 `main.py`（按下 Run）即可一次执行四种算法：
  - `GCCS-2Phase`（你的两阶段算法，通过 `two_phase_scheduler.run_scheduler` 或上传文件回退）
  - `HEFT`
  - `HydraLike`
  - `MRSALike`
- CSV 放到 `./input/segments.csv`、`./input/edges.csv`
- 输出写入 `./output/data`。

## 目录
- `main.py`：单入口（可在 IDE 里直接运行）
- `common.py`：复用模块
- `algo/`：各算法实现（two_phase/heft/hydra_like/mrsa_like）
