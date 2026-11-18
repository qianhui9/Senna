# Experiment 3：连续语义查询的时序冗余

这是上传任务对应的新入口。旧的对象特征实验 `run_exp3.py` 及其配置和结果保留。
新配置位于 `config.yaml` 的 `semantic_queries` 节，结果位于
`outputs/motivation_exp3/semantic_queries/`，避免覆盖旧实验报告。

## 手动运行命令

自动等待任务已按用户要求取消。自行确认显卡空闲后，选择对应的一条命令运行。
每条命令在指定 GPU 上执行完整实验；两张卡使用独立输出目录，并支持同命令断点续跑。
入口会做一次启动前检查，不轮询等待显卡。

GPU 0：

```bash
cd /data/qh/phdCode/work4/Senna
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/data/miniconda3/envs/Senna/bin/python -m experiments.motivation_exp3.run_queries \
  --config experiments/motivation_exp3/config.yaml \
  --device cuda:0 --output-dir outputs/motivation_exp3/semantic_queries_gpu0 \
  --mode all --existing resume
```

GPU 1：

```bash
cd /data/qh/phdCode/work4/Senna
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/data/miniconda3/envs/Senna/bin/python -m experiments.motivation_exp3.run_queries \
  --config experiments/motivation_exp3/config.yaml \
  --device cuda:1 --output-dir outputs/motivation_exp3/semantic_queries_gpu1 \
  --mode all --existing resume
```

默认 3 个 val 场景，长度 5/10/20，每个场景和长度有五种 fixed 序列及一种 mixed 序列。可加 `--max-scenes 10` 扩大覆盖。

## 检查、Smoke 与续跑

下面的检查不加载 Senna 模型：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/data/miniconda3/envs/Senna/bin/python -m experiments.motivation_exp3.run_queries --dry-run
```

CPU 回归测试：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/data/miniconda3/envs/Senna/bin/python -m unittest \
  experiments.motivation_exp3.tests.test_semantic_queries \
  experiments.motivation_exp1.tests.test_profiling_utils -v
```

真实数据和模型 Smoke（独立输出 `semantic_queries_smoke/`）：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/data/miniconda3/envs/Senna/bin/python -m experiments.motivation_exp3.run_queries \
  --mode all --smoke-test --existing resume
```

只做离线 GT 分析使用 `--mode data`；已有完整构建后，可用 `--mode profile` 测量、
`--mode report` 重建统计和图表。`data` 不会提供虚构 GPU 时延。
`--device cuda:0`、`--checkpoint-path`、`--dataroot`、`--output-dir`、
`--sequence-lengths 5 10 20`、`--warmup-iterations`、`--profile-repeats` 可覆盖 YAML。
也支持 `SENNA_MODEL_PATH`、`NUSCENES_ROOT`。

默认按同配置续跑：逐请求 JSONL 在写入完成、flush/fsync 后成为断点，恢复时丢弃中断的末行、
拒绝重复 ID 和校验失败记录；已完成正式测量无需再加载模型。输出目录有进程锁。
源码 SHA256、完整解析配置、模型和官方数据文件的大小/mtime 共同构成身份；
完成的构建及推理产物另有 SHA256 校验。更改配置/源码/输入后使用新目录，或
`--existing overwrite` 归档本入口自己的结果后重算。不会覆盖没有本入口 manifest 的旧文件目录。

## 查询与真实 Ground Truth

- **Category/Position**：在当前 LiDAR 坐标（x 右、y 前）的固定前方扇区和半径中，
  选择满足传感器点数条件的最近车辆；排除自行车、摩托车，以 instance token 打破距离平局。
  Category 的答案是官方 `category_name`；Position 由配置的横向边界和标签确定。
  每帧重新选择，显式记录目标切换。
- **Existence**：沿用原始 VRU Identification 问题结构，检查官方标注中的行人、自行车、
  摩托车是否位于固定作用域。记录整个成员集合，包含低点数但存在的官方实例。
  空集是真实 NO；缺失/损坏标注及缺图均有过滤原因，不能转成 NO。
- **Motion Prediction**：沿用原始单对象 SPEED/PATH 任务、动作标签函数和对象局部坐标变换，
  从 `sample_annotation.next` 提取同实例六步未来轨迹。要求每步对应紧邻真实 sample，
  时间间隔在显式容差内。原始 QA 代码从未来位移估计输入速度，这里改为严格使用过去
  annotation 的后向差分，缺少过去或未来标注均过滤。当前位置和速度是 GT 条件输入，
  不是模型自己检测的结果。未来位置、时间戳、annotation token 仅存入离线标签字段。
- **Meta-action Planning**：复用 `planning_question()`、`current_speed()`、
  `navigation_command()`、`parse_plan()` 和原始 SPEED/PATH 标签算法。
  原始六秒未来路线来源保留为 `offline_future_route_6s` 导航条件；三秒未来轨迹只生成 GT。
  这不是在线无未来信息规划测试。图像遵循共享输入协议的六视角顺序。

所有范围、类别前缀、传感器点数、位置边界、速度区间和未来轨迹间隔都在 `config.yaml` 中固定。
对象/场景/sample/annotation 身份和图像关联保留。`model_sample()` 用显式字段白名单构造请求，
不会传入 `ground_truth`、依赖信息或 `offline_ground_truth`。

## 连续序列与统计口径

只沿同场景的 `sample.prev/next`，检查链完整性与严格递增的真实时间戳。
每场景从第 1 帧开始取固定窗口，保留第 0 帧作为后向速度观测。
`fixed` 是固定问题模板：状态查询的文本也固定；运动的当前位置/速度及规划条件会更新。
`mixed` 按五种模板循环。过滤后不把不相邻帧连成有效 pair。

答案稳定性与依赖稳定性独立统计，尤其标记答案相同但对象/事实已变的 pair。
Category 签名为实例+类别；Existence 为作用域+实例集合；Position 为实例+位置标签；
Motion 为实例+当前运动区间+预测标签；Planning 为导航+当前速度输入/区间+规划标签。
同时报告 Jaccard、目标切换、连续位置/速度变化及离散标签改变。
两个空的 Existence 集合 Jaccard=1；无对象集合的 Planning 不填 Jaccard=1。

相邻同模板主统计从 `base_queries.json` 去重，防止 5/10/20 的重叠前缀重复计数。
CSV 同时给出 pooled pair 比例、scene 等权均值、scene Bootstrap 95% CI。
重复测量先在请求内平均，推断的重采样单位始终为 scene；少于两个 scene 不提供 CI。
具体工作负载转移和无历史事件单独统计，不混入 Answer Stability 的分母。

## Oracle 不代表运行时能力

只检查紧邻上一时刻请求暴露的事实，不建立完整累积世界状态。
当前所需事实必须齐全，历史事实值必须仍有效，并能得到当前正确答案，才计为安全机会。
最近目标身份、Existence 的作用域和整个实例集合均需要稳定。
缺少事实类型/目标、stale answer、依赖改变和 safe 分别记录；这些分类可重叠。

Motion/Planning 的 Oracle 使用离线历史 GT 预测/规划标签，属于包含未来衍生标签的乐观上界，
不能解释为只凭可观测当前世界事实就能在线回答预测问题。
离散签名稳定不代表连续世界状态完全不变，连续差异会另外保存。
WorldStream/QCSE 的实际状态维护和复用尚未实现。本实验不实现事实直答、答案缓存、
跨请求 Token/KV 复用，不把 Oracle 指标包装成实测加速。

## Profiling 与交付

直接调用 Experiment 1 的 `inference_args()`、`original_request()`、`timed_request()`，
复用同一原始模型 loader、预处理、generation 参数、CUDA Event/Hook 和 Prefill/Decode 拆分。
不另外实现计时系统，也不修改 Senna。每个类型验证一次原始/插桩输入摘要及输出 token/回答完全等价。
Warmup 和等价性请求均不进入正式数据。Prefill 包含首 token，Decode 为之后的缓存 forward；
generation overhead 与视觉/I/O 成本属于 E2E 的其他部分。无效分解保持缺失值。
图中的分解使用同时拥有 E2E/Prefill/Decode 的一致请求群体。

产物：`query_traces.csv`、`query_dependencies.csv`、`dependency_transitions.csv`、
`query_profile_samples.csv`、`query_type_summary.csv`、`sequence_summary.csv`、
`aggregate_summary.json`、`experiment_report.md`。
额外保存查询 JSON、过滤记录、配置/身份 manifest、模型元数据、逐类型等价性和验证记录。
`dependency_transitions.csv` 通过 `comparison` 区分去重主统计和实际工作负载转移。
七组图位于 `figures/`，各有 PNG 300 DPI 和 PDF 矢量版本；未测得的数据不会伪造。
完整序列成本分别测量并累加；过滤导致的不完整序列明确标记，累计成本图只使用标注完整的序列。

真实产物验证（只读，不加载模型）：

```bash
SENNA_EXP3_QUERY_OUTPUT=outputs/motivation_exp3/semantic_queries \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/data/miniconda3/envs/Senna/bin/python -m unittest \
  experiments.motivation_exp3.tests.test_real_semantic_queries -v
```

这六项验证检查六视角关联、现有 corrected 官方规划 QA 的输入/GT 一致性、
未来轨迹只在离线标签中出现、主 pair 去重与 Oracle 前提、七组图格式，以及正式结果排除共享负载。
旧对象特征测试依赖 `pandas`，不属于本入口的最小依赖；当前 Senna 环境缺少它时，
旧特征测试的导入失败需要单独记录，不应当作新语义入口测试通过。

本次实际执行记录位于 `outputs/motivation_exp3/semantic_queries/validation.json`。
自动等待会话 `senna_exp3_queries_idle` 已按用户要求取消。正式 GPU 测量由用户手动启动；先前共享负载的完整 Smoke 仅用于验证推理链路，未合入正式统计。
