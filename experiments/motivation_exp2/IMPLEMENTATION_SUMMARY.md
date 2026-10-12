# 第二组 Motivation Experiment 实现与验收记录

## 1. 检查和复用的现有内容

检查了 Senna 运行文档、项目依赖及 `experiments/motivation_exp1/` 的配置、索引、统计和绘图模块。复用第一组 `config.yaml`、`outputs/motivation_exp1/resolved_config.yaml` 的数据路径、`manifest_summary.json` 的实际场景列表、`camera_manifest.csv` 的关键帧记录，以及 `utils/data_utils.py` 的官方加载器/六相机协议/场景选择/原子输出和 `utils/stats_utils.py` 的场景 Bootstrap。未使用 sweep pair 作为带实例标注的物体对。

第一组 44 个输出文件哈希未变化。

## 2. 新建和修改的文件

新建文件如下；另修改项目 `运行文件.md`，增加 CPU 实验命令。

- `experiments/motivation_exp2/README.md`
- `experiments/motivation_exp2/__init__.py`
- `experiments/motivation_exp2/aggregate_results.py`
- `experiments/motivation_exp2/analyze_ego_motion.py`
- `experiments/motivation_exp2/analyze_grid_overlap.py`
- `experiments/motivation_exp2/analyze_image_displacement.py`
- `experiments/motivation_exp2/analyze_physical_persistence.py`
- `experiments/motivation_exp2/build_keyframe_pairs.py`
- `experiments/motivation_exp2/config.yaml`
- `experiments/motivation_exp2/extract_projected_annotations.py`
- `experiments/motivation_exp2/gen_M2a.py`
- `experiments/motivation_exp2/gen_M2b.py`
- `experiments/motivation_exp2/gen_M2c.py`
- `experiments/motivation_exp2/gen_M2d.py`
- `experiments/motivation_exp2/gen_M2e.py`
- `experiments/motivation_exp2/generate_report.py`
- `experiments/motivation_exp2/plot_results.py`
- `experiments/motivation_exp2/requirements.txt`
- `experiments/motivation_exp2/run_exp2.py`
- `experiments/motivation_exp2/tests/__init__.py`
- `experiments/motivation_exp2/tests/test_configuration.py`
- `experiments/motivation_exp2/tests/test_displacement.py`
- `experiments/motivation_exp2/tests/test_grid_overlap.py`
- `experiments/motivation_exp2/tests/test_instance_matching.py`
- `experiments/motivation_exp2/tests/test_pair_index.py`
- `experiments/motivation_exp2/tests/test_projection.py`
- `experiments/motivation_exp2/tests/test_real_nuscenes_regression.py`
- `experiments/motivation_exp2/utils/__init__.py`
- `experiments/motivation_exp2/utils/annotation_utils.py`
- `experiments/motivation_exp2/utils/data_utils.py`
- `experiments/motivation_exp2/utils/geometry_utils.py`
- `experiments/motivation_exp2/utils/statistics_utils.py`
- `experiments/motivation_exp2/utils/visualization_utils.py`

## 3. 配对和实例身份

沿 scene.first_sample_token 和 sample.next 配对，gap=1/2/4，禁止跨场景；保留实际 sample 和各路相机曝光时间。只将同一场景、同一相机、相同官方 instance_token 且双帧有效投影的对象纳入位移统计。Annotation token 随时刻变化，不充当物体身份。

## 4. 几何投影和运动定义

使用官方 get_sample_data(..., BoxVisibility.NONE) 获取相机坐标 Box，再用 view_points 投影角点。每帧使用自己的曝光时刻 Ego Pose、标定和真实图像尺寸；保留原始和边界裁剪后的矩形。近裁剪面交叉明确排除；默认最小宽高 8px、几何矩形在图像内面积比例≥0.2。该比例不是精确遮挡可见性。关键帧标注不在相机曝光时间做插值，记录并说明时间偏差。

列向量约定下，T_rel=inverse(T_global_ego0)@T_global_ego1；平移分量位于初始自车坐标系，旋转用归一化四元数最短角，Yaw 差处理 ±π 跨越。运动速率用实际 camera 时间差；物体速度用全局标注中心和实际 sample 时间差。

位移为裁剪矩形中心欧氏距离，除以实际图像对角线归一化；IoU 比较同一路相机固定像素坐标下的跨时刻矩形。诊断网格采用任意正面积交集、最小 cell 面积覆盖比例和 cell 中心三种规则；分别输出 Jaccard 和旧格保留率。

## 5. 实际运行与产物

- nuScenes v1.0-trainval：10 场景、403 关键帧、1139 个关键帧对。
- 有效投影 19390 次；同实例双帧相机观测 40043 条，涉及 1033 个不同实例。
- 42 项测试全部通过，无失败、错误或跳过；真实回归覆盖 60 个相机视角。
- dry-run、真实 smoke、完整实验、独立 report 阶段、全阶段结果复用均已执行通过。
- 七张要求的原始 CSV、额外帧/标注/位姿索引、分类/相机/运动/距离/场景统计、1,000 次场景 Bootstrap 和阈值敏感性分析已生成。
- M2(a–e) 与四页定性案例：8 张主图 PNG 和 8 份 PDF；30 组独立真实案例，12 张官方 3D 线框回归图，共 50 张 PNG，均为 300 DPI。照片在 PDF 中为嵌入栅格，统计图和文字为矢量。
- 结果位于 `outputs/motivation_exp2/`；smoke 位于 `outputs/motivation_exp2_smoke/`。较早的生成版本保留在各结果目录的 `previous_runs/`。

## 6. 真实观察与限制

| Gap / 约秒数 | 全局标注持续率：场景均值 | 归一化位移：观测中位数 | 16×16 Jaccard：场景均值 |
| --- | --- | --- | --- |
| 1 / 0.5s | 0.9531 | 0.0350 | 0.5138 |
| 2 / 1s | 0.9141 | 0.0605 | 0.3640 |
| 4 / 2s | 0.8399 | 0.0887 | 0.2630 |

这些测量支持所选真实场景中，物理实例身份的持续不保证固定图像坐标和网格对应区域稳定。Ego-motion 相关性按静止/运动对象、时间跨度、距离和相机进行分层；场景级相关区间较宽，不能断言自车运动是唯一因果来源。

投影矩形不是对象分割或精确可见性；存在标注/曝光时间差和边界裁剪影响。静止标注目标不等于所有静态背景；10 场景不代表整个驾驶总体。网格重叠不等于 VLM Cache 命中率；本实验不建立任何模型精度、推理速度、计算跳过或调度收益结论。未实现任何 WorldStream 方法、模型训练或视觉特征复用。

完整指标、置信区间、失败/排除状态、支持的观察与未建立的结论见 `outputs/motivation_exp2/experiment_report.md`。

## 7. 一条命令复现

在 Senna 项目根目录执行：

```bash
/home/qh/miniconda3/envs/Senna/bin/python experiments/motivation_exp2/run_exp2.py --config experiments/motivation_exp2/config.yaml --run-all
```

同一代码、配置和数据会验证文件哈希并复用结果；加 `--existing overwrite` 可重新测量全部阶段。所有 YAML 参数都可通过 `--set section.key=YAML` 覆盖。

图表工作流采用 `/home/qh/.codex/skills/paper-figure/SKILL.md`，已完成实际渲染检查和独立图表审阅。
