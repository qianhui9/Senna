# Experiment 3: Physical Identity vs. Representation Stability

使用第二组 nuScenes 官方实例和原始 Senna 完整视觉编码，测量同一物体在不同观测中的表示变化。没有训练、特征替换、在线缓存、增量编码或任何 WorldStream 方法。默认环境为 `Senna`，无需增加依赖；测试使用 Python 自带的 `unittest`。

## 运行

在项目根目录执行：

```bash
/home/qh/miniconda3/envs/Senna/bin/python experiments/motivation_exp3/run_exp3.py --dry-run
/home/qh/miniconda3/envs/Senna/bin/python -m unittest discover -s experiments/motivation_exp3/tests -v
/home/qh/miniconda3/envs/Senna/bin/python experiments/motivation_exp3/run_exp3.py --max-scenes 2 --max-object-pairs 100 --smoke-test
/home/qh/miniconda3/envs/Senna/bin/python experiments/motivation_exp3/run_exp3.py --config experiments/motivation_exp3/config.yaml --run-all
```

`--dry-run` 检查配置、现有 E2 输入、层索引、权重分片和 processor，无 GPU 前向或输出写入。Smoke 使用独立的 `outputs/motivation_exp3_smoke/`。完整运行默认 10 scenes、3000 对；采用 E2 scene 名称排序的前 N 个 scene，所有场景、相机及筛选比例记录在报告中。

模型路径优先级：`--checkpoint-path` / 配置 → `SENNA_MODEL_PATH` → E1 的 `outputs/motivation_exp1/resolved_config.yaml`。数据根目录同理使用 `--dataroot` / `NUSCENES_ROOT` / E1 配置。未硬编码权重或数据绝对路径。实际模型/processor 使用本地权重。完整图像必须存在；不要用测试生成的随机权重或图像作为研究证据。

```bash
python experiments/motivation_exp3/run_exp3.py --run-all \
  --checkpoint-path /path/to/original/Senna --dataroot /path/to/nuscenes \
  --exp2-output-dir outputs/motivation_exp2 --output-dir outputs/motivation_exp3_custom \
  --device cuda:0 --batch-size 1 --feature-layers 0 6 12 18 24 \
  --set analysis.bootstrap_iterations=1000 --set analysis.seed=42
```

所有已定义配置可由 `--set SECTION.KEY=YAML` 覆盖。为保持原始 loader 的行为，只允许原始 fp16、原始预处理和 inference_mode。GPU OOM 首先拆成单图重试；单图仍失败时记录并停止，不生成空成功结果。S2 / anyres 配置明确拒绝，必须扩展逐尺度/逐 tile 映射后才能支持，不能当成单尺度规则网格。

## 分阶段和断点续跑

```bash
python experiments/motivation_exp3/run_exp3.py --stage build-pairs
python experiments/motivation_exp3/run_exp3.py --stage extract-features
python experiments/motivation_exp3/run_exp3.py --stage map-patches
python experiments/motivation_exp3/run_exp3.py --stage object-features
python experiments/motivation_exp3/run_exp3.py --stage similarity
python experiments/motivation_exp3/run_exp3.py --stage layer-analysis
python experiments/motivation_exp3/run_exp3.py --stage temporal-analysis
python experiments/motivation_exp3/run_exp3.py --stage static-dynamic
python experiments/motivation_exp3/run_exp3.py --stage aggregate
python experiments/motivation_exp3/run_exp3.py --stage plot
python experiments/motivation_exp3/run_exp3.py --stage report
```

同一配置直接重复 `--run-all` 即可续跑：配置、源码、E2 文件 SHA256、模型文件大小/mtime 和逐图分片 SHA256 一致才接受已有结果。上游重跑会失效下游完成标记；被破坏的特征分片会重新完整编码。改变输入/源码/配置时使用新输出目录，或 `--run-all --existing overwrite` 将旧结果归档到 `previous_runs/` 后重算。写出的特征和统计文件采用原子替换。

各模块也可通过 `python -m experiments.motivation_exp3.<模块> --stage ...` 调用。没有 prerequisite 的单独阶段会明确报错，不会补入假数据。

## 本地原始模型和预处理

当前 checkpoint：`CLIPVisionTower`，ViT-L/14，24 blocks，hidden dimension 1024，336×336 输入，24×24=576 patches，额外 1 个 CLS。Hidden-state 0 是加入位置编码后经过 pre_layernorm 的 embedding，hidden-state i 是第 i 个 block 后的输出。Senna `mm_vision_select_layer=-2` 对应 25 个 hidden states 中的索引 23；默认自动补入 23，并保留 0、6、12、18、24。

原始 `SennaLlavaLlamaForCausalLM.encode_images()` 只运行 tower；`mm_project()` 逐视角运行 MLP2xGELU (1024→4096→4096)，随后 `img_adapter` 在 token 轴上全局混合 576→128→128→128。Projector 前后空间一一对应；128-token adapter 不具有物体 patch 索引，只分析全图均值。运行时记录实际张量，不把配置推断写成未经验证的输出。

代码复用 `llava.mm_utils.process_images`。nuScenes 1600×900 图像采用均值颜色补成 1600×1600（上/下各 350），然后缩放、center crop 至 336×336；bbox 坐标平移后按真实整数 resize 尺寸缩放。实现也覆盖仅 resize、非方形图像、center crop 和 crop 的零 padding，逐图验证张量输入尺寸，保留裁剪范围/面积。Normalize 不改变坐标。

只读 forward hooks 从原始 `CLIPVisionModel(output_hidden_states=True)` 输出读取层，不改变返回值；对原始 tower 输出和整个压缩视觉输出进行数值等价比较，另重复相同图像得到数值控制。CLS 在 spatial pooling 前明确去除；所有层使用同一组 bbox/patch 索引。

为限制显存和存储，提取阶段先在 CPU 计算逐图映射并立即聚合对象描述符，保存一个离线 `.npz` 分片；`map-patches` 阶段再发布完整映射 CSV，`object-features` 校验并索引描述符。这是融合的离线统计实现，不需要将所有图像 hidden states 先落盘。可选 `output.save_raw_features=true` 保存上限和预算内的原始空间张量用于调试。分片存储 area/mean/center 描述符、预先定义的 4×4 固定 patch 中心采样及全图均值，含 scene/camera/timestamp/instance 元数据，不含推理缓存。

## 真实身份与对照

复用 E2 `data/keyframe_pairs.csv`、`projected_objects.csv`、`camera_frames.csv` 和 `ego_motion_displacement.csv`，缺少合并位移表则复用 `image_displacement.csv` + `ego_pairs.csv`；缺少位移表则调用 E2 原有 `compute_rows`。无须重新加载整个 nuScenes devkit 表或编写跟踪器。原始投影、官方 instance_token、类别、相机/样本 token 和 bbox 都进行交叉检查。不可靠或不存在字段保留 NaN。

- B1：同一官方 instance、同 scene/camera、正向真实时间差。
- B2：同类别不同 instance；与 B1 共享 anchor、scene、camera 和 exact target frame。
- B3：不同类别不同 instance；其他匹配条件同 B2。
- B4：相同图像 patch 坐标跨帧；均匀预定 4×4 patch 中心采样，剔除 padding 中心，统计先按 frame pair 平均。无物理身份要求。

不同 ID 的候选来自选中帧中所有有效官方投影，包括未持续存在的物体；采用预先声明的目标框面积和可靠距离比例约束（默认均 ≤2 倍），在通过条件的候选中确定性随机选择。没有满足条件的负样本时记录 unmatched，不放宽标准。负例可能重复用于多个 anchor，因此它们不是独立实验单位。M3(a) 分别展示 B1/B2 和 B1/B3 的 matched anchor cohort，避免把不同的样本总体混在一起解释 identity margin。

面积权重是 bbox 与 patch 相交面积，默认保留 intersection/patch area ≥0.25 的 patches。Mean 对同样的 patches 不加权；center 只选择中心在框内的 patches。对象描述符包含背景和全局上下文，不是模型原生 Object Token。

## 指标、统计与输出

FP32 cosine similarity / distance 和 relative L2（分母 `norm(t1)+1e-8`）；非有限值、近零范数、无有效 patches 分别记录。报告均值、标准差、P25/P50/P75/P90/P95、n、scene-equal mean 及 scene bootstrap CI。一个 scene 不生成虚假的 CI。

时间分组使用真实 camera exposure timestamp，approximate speed 仍沿用 E2 annotation sample timestamp。支持位移、ego translation/rotation/yaw、对象面积/尺度、可靠距离和几何裁剪质量分层；静态/运动对象在类别和时间跨度内部比较，速度阈值 0.2/0.5/1.0 m/s 敏感性；patch 数量至少 2/4/8、面积、三种聚合策略敏感性。各层只比较所有提取层均有效的同一 cohort。

Matched ΔS、分布直方图 overlap 和描述性 AUC 只用于离线观察，不训练分类器，不设复用阈值。Scene bootstrap 处理重复物体、共享帧、相机和负例相关性；Spearman 是描述性关联，无 IID p-value，不作因果解释。分层随机抽样刻意保留不同条件，结果不是 nuScenes 总体估计。

`outputs/motivation_exp3/` 下保存：

- `metadata/`：resolved config、原始模型/processor、运行等价和重复输入检查、scene、过滤/配对/提取状态及哈希。
- `data/`：对象对、观测、patch 映射、特征索引、三类对象 similarity、独立 B4/global similarity、匹配负例、layer/temporal/motion/condition/sensitivity/margin 表。
- `features/object_feature_shards/`：逐图 offline object descriptors；默认无全图原始层 dump。
- `figures/`、`figure_pdf/`：M3(a)–M3(f) 六类 PNG/PDF。
- `visualizations/`：官方 bbox 对应示例。四个示例条件取分位数内的中心案例，另加固定 seed 参考样本，不只挑支持假设的极端案例。
- `statistics/summary.json`、`condition_correlations.json` 和 `experiment_report.md`：真实数值、CI、未匹配与过滤数量、结论和研究边界。

## 正确性验证

测试覆盖跨 scene/camera/ID 禁配、bbox/身份元数据矛盾、排序/重复、missing image、pad/resize/crop/边界/完全裁剪、实际 processor 像素标记、真实 HF tiny CLIP 的层/CLS/batch/重复输入/输出等价与参数不变、FP16→FP32、相交面积和零权重、原子分片/元数据、scene CI 和 hash resume。Tiny 随机模型只用于单元测试，与真实实验输出隔离。

结论只能描述现有 Senna 的观测表示；不能推导缓存必错、驾驶任务准确率变化、最优阈值、World Alignment/State Validity 有效性、FLOPs/GPU 加速或 scheduler 必要性。
