# Experiment 2: Spatially Non-uniform Visual Feature Changes

这是本次真实视觉特征实验的入口。已有 `run_exp2.py` / `config.yaml` 属于原来的几何持续性实验，保留使用；新实验使用 `run_feature_drift.py` / `feature_drift_config.yaml`。

从仓库根目录运行：

```bash
conda run --no-capture-output -n Senna python -m experiments.motivation_exp2.run_feature_drift \
  --config experiments/motivation_exp2/feature_drift_config.yaml \
  --exp1-directory outputs/motivation_exp1 \
  --checkpoint-path /data/qh/phdCode/work4/models/Senna \
  --device cuda:0 --output-directory outputs/motivation_exp2 --existing resume
```

Smoke test 用相同命令加 `--smoke-test --output-directory outputs/motivation_exp2/feature_smoke`（2 scenes、36 对，覆盖 6 cameras × 3 gaps）。`--dry-run` 验证真实索引及六路曝光补齐，不加载模型。

`--mode extract|analyze|plot|report` 支持分阶段执行；默认 `all`。`--existing resume` 校验配置、源代码、权重文件元数据、输入索引和逐对距离 SHA256。若改变配置/代码，使用新目录或 `--existing overwrite`；旧结果会归档到 `previous_runs/`。保留原几何实验结果，原报告在首次写入特征实验报告前归档。

复用 E1 的 `camera_manifest.csv` 和 `temporal_pairs.csv`，缺失时委托 E1 的 index 阶段生成，不扫描或重新配对 nuScenes。输出 `selected_temporal_pairs.csv` 是已有 pairs 的平衡子集。非关键帧的其他摄像头以相同 scene 内最近曝光补齐六路，目标摄像头严格使用 pair 中的精确端点；偏移逐对记录，超限报错。

模型使用原始 `load_senna_pretrained_model`，强制确认 `SennaLlavaLlamaForCausalLM`。复用原始预处理，执行原始 `prepare_inputs_labels_for_multimodal`，通过 Tower hook 抓取实际特征；Projector 和 Image Adapter 也实际执行。首个六路输入与原始生成 helper 验证预处理哈希和所有 Tower 输出逐元素完全相等。CLIP patch convolution 的实测行列数、原始 CLS 结构与 embedding 展平顺序共同确认网格；不靠 Token 数开平方猜布局。

分析的是 Senna 实际使用的 Tower 输出。当前 checkpoint 选择 `hidden_states[-2]`，因此不能将其误写成 CLIP 最后 Transformer block 的输出。所有实测形状、层索引和原始 helper 等价验证保存在 `model_metadata.json`。

每对仅暂存两个目标摄像头的 CPU 特征，算完距离即释放；不存储特征历史，不复用跨帧特征。断点文件只保留距离及 provenance。零范数、非有限数据、缺失图像和异常配对会记录到 `feature_issues.json`，缺失对也在 samples 表输出 failed 行，进程返回非零。分析命令可以汇总已成功的真实部分结果。

余弦距离为 `1-cos`，辅助 L2 使用每个 Token 向量单位归一化后的欧氏距离。Stable 比例使用所有预设阈值。Top-K 使用 ceil(N*k)，图中显示实际 k/N 均匀基线。Gini 衡量幅度不均匀，不代表空间邻接性；热力图和 3×3 区域统计提供位置证据。全部网格位置均对应实际 padded/resized 模型输入，未做物体运动补偿。

额外报告 `content_*` 敏感性指标：仅统计前后两帧均有 ≥50% 几何面积与真实图像重合的 Patch，以及完全 padding 区域的平均漂移。该子集不改变任何模型计算，避免将 padding 引起的稳定性过度归因于道路内容；不精确建模插值边界。1600×900 输入在 24×24 网格中有 336 个内容主导 Patch、240 个完全 padding Patch。补齐相机容差为 100 ms（现有 E1 配对最大偏移为 92.384 ms），目标摄像头端点始终保持精确匹配。

统计先在 pair 内计算、scene 内平均，再等权平均 scenes；95% Bootstrap CI 只重采样 scenes。分位数汇总是每对 Token 分位数的平均，不是所有 Token 的混合分位数。CDF 是 scene 等权经验分布，CI 为逐点区间。小于两个 scene 不给出 CI。

Profiling 仅在 E1 的权重、模型/预处理代码、配置、GPU 型号/设备、依赖和实测输入形状全部匹配时复用；否则 warmup 后补测少量 CUDA Event 样本。E1 未记录物理 GPU UUID，硬件匹配只能达到已有记录的精度。六视角成本已包含六路，不能再乘六。稳定比例只表示理论机会，不等于可省 FLOPs 或真实加速。

输出包括用户要求的八项 CSV/JSON/报告，以及逐对距离、scene/区域统计、CDF 曲线、profiling provenance 和八组 PNG 300 DPI / 矢量 PDF 图。

测试：

```bash
conda run --no-capture-output -n Senna python -m unittest experiments.motivation_exp2.tests.test_feature_drift -v
SENNA_FEATURE_DRIFT_OUTPUT=outputs/motivation_exp2 \
  conda run --no-capture-output -n Senna python -m unittest experiments.motivation_exp2.tests.test_feature_drift -v
```

单元测试的已知向量仅用于数学正确性检验，实验输出全部来自真实模型。
