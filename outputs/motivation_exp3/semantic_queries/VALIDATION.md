# Experiment 3 验证与运行状态

语义查询实现与离线分析已完成。正式测量当前未完成，用户将自行在 GPU 空闲时手动启动，不能把共享负载的 Smoke 时延当作正式结果。

- 25 项语义查询测试、11 项复用 profiling 测试通过（针对性测试共 36 项）。
- Experiment 1 完整回归 38 项通过；6 项真实产物验证通过。其中 11 项 profiling 测试与上述回归重叠。
- Fresh agent 原样运行文档 dry-run 和 CPU 回归，exit code 均为 0。
- 真实原始 Senna Smoke：2 场景、12 序列、42 次查询，五种类型输入/输出等价性全部通过，42 项时延分解有效。仅作为共享 GPU 下的链路验证，未合入正式测量。
- 正式离线结果：3 场景、54 序列、436 个有效事件、228 个不同 sample/模板、212 个去重相邻同模板 pair。
- 旧特征实验测试因当前最小推理环境缺少 pandas，有 5 个模块导入失败；新语义入口不依赖 pandas。

完整证据见 [validation.json](validation.json)。正式测量进度见 [query_invocation_status.json](query_invocation_status.json)，运行完成后报告与图表会自动更新。

自动等待会话 `senna_exp3_queries_idle` 已按用户最新要求取消，当前没有后台 GPU 轮询。

完整运行与恢复命令见 [SEMANTIC_QUERIES.md](../../../experiments/motivation_exp3/SEMANTIC_QUERIES.md)。
