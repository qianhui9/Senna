# Experiment 3 — Temporal Redundancy in Semantic Query Execution

状态：`offline_or_partial_profile`。采样 3 个官方 val 场景，54 条序列，436 次有效查询事件，228 个不同 sample/模板，212 个去重后的有效相邻同模板查询对。
原始 Senna 正式测量：0/436；warmup 与等价性请求不计入。

## 构建与真值来源

- Category/Position：官方 sample_annotation，以当前 LiDAR 坐标筛选前方扇区内最近车辆，按 instance token 打破距离平局；类别直接来自 category_name，位置由固定横向边界离散。每帧重新选择，明确记录目标切换。
- Existence：沿用 VRU Identification 问题结构，以官方标注中的行人、自行车、摩托车及固定前方空间范围判定；空集是真实负例，缺失或损坏标注不能当成负例。
- Motion Prediction：沿用原始 Motion Prediction 的 SPEED/PATH 任务与 get_obj_acc_or_dec、get_obj_turn_or_lane_change；要求同实例完整连续六步未来标注，保留未来位置、真实时间戳和标签。输入速度仅由过去标注计算，不使用原始构造代码中的未来位移速度。目标位置与速度是当前 GT 条件，非纯视觉检测结果。
- Meta-action Planning：直接调用 planning_question/current_speed/navigation_command，使用原始三秒未来自车轨迹标签及 parse_plan。原始六秒未来路线仅作为明确标记的离线导航条件，不能据此宣称在线无未来信息的规划表现。
- 全部对象、场景、sample、annotation token、六视角路径、输入协议、离散规则和连续属性可在 query_traces.csv/query_dependencies.csv/queries.json 中追溯。未来数据只存在离线标签字段，模型输入由 model_sample 的字段白名单构建。

## 统计口径

同场景 sample.prev/next 建立连续序列，取每场景第 1 个关键帧开始的固定前缀（跳过第 0 帧以支持过去速度）。固定模板与具体输入文本分别记录：运动提示的当前数值、规划的速度和导航会随帧变化。Fixed 指固定问题模板；Category/Existence/Position 的问题文本也完全相同。
相邻答案和依赖统计基于 base_queries 去重，不重复计算不同长度的重叠前缀。过滤后不跨帧连线。图中为场景等权均值，CSV 同时给出 pooled 比例和 scene Bootstrap 95% CI；少于两个场景不给 CI。

| 查询类型 | 有效相邻对 | Answer 稳定 | Dependency 稳定 | Oracle 安全机会 |
|---|---:|---:|---:|---:|
| category | 35 | 82.07% | 72.90% | 72.90% |
| existence | 57 | 94.74% | 94.74% | 94.74% |
| position | 35 | 98.25% | 72.90% | 72.90% |
| motion_prediction | 28 | 85.19% | 67.59% | 67.59% |
| meta_action_planning | 57 | 92.98% | 75.44% | 75.44% |

## Oracle 的含义与边界

Oracle 仅查看上一时刻查询暴露的事实，检查当前所需事实是否齐全、历史依赖仍有效且可得到相同答案。Existence 要求作用域及实例集合完全一致；Category/Position 同时验证最近目标身份。数值位移、速度差和离散属性变化另行保留。Mixed 只使用紧邻上一条查询的事实，不累积整段完整世界状态。
Motion/Planning 的 Oracle 使用离线可知的历史预测/规划 GT 标签，因此是包含预测标签的乐观 GT 上界；这些标签不是可在线观测的当前环境事实。稳定的离散签名也不表示所有连续属性完全不变。不能把这类机会等同于可部署的事实复用。
分类计数如下（不是互斥分类；依赖改变可能同时发生旧答案错误）：

```json
{
  "oracle_safe_reuse": 163,
  "stale_answer": 7,
  "dependency_change": 49,
  "state_information_unavailable": 31
}
```

Oracle、原始 Senna 回答和 WorldStream 能力严格分开：本实验未实现 QCSE、事实直答、答案缓存、历史视觉 Token/KV 复用或运行时世界状态维护。所有正式请求仍执行原始 Senna；Oracle 不是实测加速收益。

## 原始 Senna 实测

| 查询类型 | 请求数 | Prefill ms | Decode ms | E2E ms | 输出 token | 严格正确率 |
|---|---:|---:|---:|---:|---:|---:|
| category | 0 | 未测得 | 未测得 | 未测得 | 未测得 | 未测得 |
| existence | 0 | 未测得 | 未测得 | 未测得 | 未测得 | 未测得 |
| position | 0 | 未测得 | 未测得 | 未测得 | 未测得 | 未测得 |
| motion_prediction | 0 | 未测得 | 未测得 | 未测得 | 未测得 | 未测得 |
| meta_action_planning | 0 | 未测得 | 未测得 | 未测得 | 未测得 | 未测得 |

Prefill/Decode 复用 Experiment 1 CUDA forward 区间：Prefill 产生首 token，Decode 为后续带 KV 的单 token forward。CUDA 区间包含 dispatch/stream 空闲，不是纯 kernel 时间；generation overhead、图像编码、I/O 等包含在 E2E 的其他部分。每个真实序列分别执行，不从其他长度或负载借用计时。
每个序列的成本只在所有有效请求均完成测量时求和；注释缺失造成的不完整序列明确标记，累计成本图仅展示 annotation_complete 的序列。真实预测、标签和输出 token 存于 query_profile_samples.csv。

硬件负载、版本、模型配置与原始 generation 参数见 query_model_metadata.json；逐类型原始/插桩输入哈希、输出 token 和回答等价性见 query_output_equivalence.json。
本次测量条件：preflight requires <500 MiB used and <5% GPU utilization; no ongoing exclusive reservation。

## 过滤与验证

```json
{
  "invalid_future_object_link_or_interval": 6,
  "no_deterministic_vehicle_target": 66
}
```

测试执行记录单独保存在 validation.json（如已执行）。Smoke 数据位于独立输出目录，不能替代正式结果。

## 直接运行

```bash
cd /data/qh/phdCode/work4/Senna
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/data/miniconda3/envs/Senna/bin/python -m experiments.motivation_exp3.run_queries \
  --config experiments/motivation_exp3/config.yaml --mode all --existing resume --wait-for-idle
```

采样参数见 resolved_query_config.yaml。完整交付包括八种要求的 CSV/JSON/报告，以及七组 PNG 300 DPI 和 PDF 图。

## 本次实际验证与测量状态

36 项针对性测试、38 项 Experiment 1 完整回归（含前述 11 项 profiling 测试）、6 项真实产物检查通过。完整记录见 [validation.json](validation.json)。

自动等待任务已按用户最新要求取消；正式时延测量由用户发现 GPU 空闲后手动启动。共享负载的 42 次 Smoke 只作链路验证，未合入本报告的正式推理统计。当前正式 Prefill/Decode/E2E 仍未测得，完成后本报告会自动重建。
