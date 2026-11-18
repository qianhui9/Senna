# Experiment 1 实现与结果审计（2026-11-17）

## 结论与已有实现

本次保留原有 Experiment 1 目录、索引基础设施、原始推理入口、Hook/CUDA Event
框架、模型权重及主要产物命名。修改前 30 项测试通过，修改后 38 项测试通过。
原完整实验确实包含真实 GPU 数据；因此只重算受配对规则和展示影响的分析，复用
经验证的原始模型计时，并另做真实端到端 Smoke Test。

| 审计项 | 原实现检查 | 本次处理 |
|---|---|---|
| 六视角真实图像索引 | 沿各 scene/camera 的 sample_data.next；检查传感器、边界、环、文件、时间戳 | 原样复用 collect_frames |
| 0.083s 配对 | 原来把配置第一项当 adjacent，未执行目标时间容差；旧选中间隔 47.417–150 ms | 显式识别 0.083s，必须为直接相机链接且在 ±35 ms 内；记录真实间隔 |
| 0.25s 配对 | 最近的严格后续时间戳，真实误差在容差内 | 保留 |
| 0.5s 配对 | 164 对中只有 19 对的两端均为关键帧，不能直接解释成相邻关键帧实验 | 限定连续相机关键帧及时间容差；新 164 对全部为相邻标注 sample 转换 |
| SSIM/PSNR/归一化像素差异 | RGB、独立分析分辨率；SSIM 独立 Gaussian 参考测试通过 | 保留公式；增加平均 Patch SSIM 汇总 |
| Patch 阈值与 8×8 网格 | 正确覆盖完整图像，含不可整除边缘；阈值可配置 | 保留；增加摄像头/时间间隔/阈值的 scene 均值分布图 |
| 全局对象持续性 | 通过官方 sample_annotation.instance_token 求交/差集；空前集为未定义 | 原样复用；增加每 scene 显式汇总及出现/消失图 |
| 高相似性与对象变化的关联 | 原先仅分别展示图像与全局对象的统计 | 新增精确双关键帧 token 关联，逐相机报告 SSIM 条件下的全局集合变化 |
| 原始 Senna Profiling | 真实 CUDA Events、同步端到端 wall time；真实输入/输出一致性；父子区间分离 | 保留测量框架和原始 run/select_samples/inference_args/matched_results 函数 |
| Prefill / Decode | 通过输入 KV cache 及实际 forward 区分；事件总和和生成 token 数有校验 | 保留；验证所有归档事件数据及 Trace |
| 不可靠分项计时 | 无效分解行仍可能进入分项汇总 | 汇总时将无效阶段设为 unavailable，保留真实端到端 wall time |
| 场景统计 | scene mean Bootstrap 正确，没有将所有帧视为独立样本 | 保留；显式输出 180 个 Scene×Camera×gap 组、10 个对象 scene 组 |
| 断点图像分析 | 只按 Patch 行数判断完整；可能保留未选中的旧 pair | 验证完整唯一网格坐标，并移除当前选择外的旧行 |
| 运行来源 | Git 受挂载目录 ownership 限制会阻塞整个运行 | 标记 Git 为 unavailable 并保留错误、源码 SHA256、配置与输入身份 |
| 图表导出 | 已有 PNG 300 DPI、PDF、匹配真实请求图 | 保留；修正文字遮挡，Image Adapter 单独列出 |

## 结果来源与复用边界

原完整结果保存在：
`outputs/motivation_exp1/previous_runs/20261117_192736_before_audit_fixes/`。
归档包含原 CSV、报告、元数据、图表及完整 CUDA Trace。`audit_baseline.json`
记录修改前源码全部匹配原记录这一事实、受保护函数 AST 哈希和 44 个原产物哈希。

复用前核实：

- 原始评测 JSON SHA256 一致；模型、CLIP、数据表的 size/mtime 与原记录一致。
- 原 Senna 推理源文件和 `utils/profiling_utils.py` SHA256 未变。
- 相机索引、对象统计和模型实际执行函数 AST 未变。
- 原 100 个请求、300 条 timing 行及 1 条详细记录均可核对。
- 分解后的不重叠阶段之和与 E2E 的最大绝对误差为 **0 ms**；Prefill/Decode
  的原始 forward 事件和汇总相符；原有 2/2 个样本 token/input/output 等价性成立。
- 原 Trace 存在 1 个 Prefill、8 个 Decode 范围及 14439 个 GPU kernel 事件。

原元数据只记录部分包版本。可核对的 torch、transformers、numpy、scipy、Pillow、
matplotlib、PyYAML 版本一致；nuScenes SDK 从 1.1.11 变为 1.2.0，不能宣称环境完全
相同。复用的原始表和 Profiling 保留原出处。新 SDK 的真实 Smoke 索引 2786 行、
对象统计 79 行，在图像路径符号链接规范化后，均与旧结果对应行完全一致。

原数据/权重身份只含 size/mtime，不具备历史完整文件内容 SHA256；审计不会将其
夸大为历史内容哈希证明。原始产物本身具备归档 SHA256。所有受配对规则影响的图像
指标、关联分析、汇总、报告与图表均在当前环境重新生成。

`reuse_provenance.json` 逐项列出复用原始文件的路径及 SHA256；
`presentation_refresh.json` 记录仅重绘布局时原始数据和数值汇总未变化。

## 实际结果

10 个 scene，13857 张索引图像（11439 张非关键帧），500 对有效图像，32000 行
Patch 数据，393 对全局标注转换。每个 gap 在 10 个 scene 内先求均值，再对 scene
均值进行 2000 次 Bootstrap。图像分析分辨率 320×192，Patch 网格 8×8，不进入
模型预处理。

| 时间组 | 对数 | 实际相机时间范围 ms | SSIM scene 均值 | 95% scene Bootstrap CI |
|---|---:|---:|---:|---|
| adjacent（目标 0.083s） | 167 | 49.855–100.093 | 0.656180 | [0.598472, 0.717001] |
| 0.25s | 169 | 249.495–251.568 | 0.559763 | [0.482862, 0.634576] |
| 0.5s（相邻关键帧） | 164 | 498.940–500.201 | 0.488585 | [0.419034, 0.557641] |

原数据的相机间隔具有异步/不规则性，0.083s 是目标值，不是对所有行实际间隔的
声明。不同时间组的选中帧对不同，因此这里只作场景总体的描述性比较。

全局 Instance Persistence 的 scene 等权均值为 **0.953115**，95% CI
**[0.937373, 0.967272]**；Jaccard 为 **0.906332**；每次相邻转换平均新增
**2.139167**、消失 **2.020000** 个标注实例。

精确关联包含 164 个 camera pair、去重后 124 个全局标注转换。
`scene-0971` 的一个 CAM_BACK_RIGHT 例子：

- sample token：`75385bf352fe4743a8cbf576b1e184fb` → `956027b87e5242ec9d9d22c8be48b7ac`。
- 相机间隔 500 ms，SSIM **0.986339**，PSNR **37.190129 dB**，归一化像素差异
  **0.005813**，所有 64 个 Patch 的 SSIM 均 ≥0.9。
- 同一转换的全局标注有 **1 个新实例**、0 个消失实例；原有 46 个实例持续存在。

这证明高相机图像相似性可以与全局标注集合变化共存。它不能确定变化实例是否在该
相机可见，也未测量同一身份对象的位置等状态变化。高阈值条件样本较少，单 scene
条件的置信区间为 unavailable，不能据此推出总体变化率。

经验证复用的完整原始 Senna 请求时延：

| 阶段 | 平均 ms |
|---|---:|
| Vision Tower | 56.269737 |
| Projector | 2.450026 |
| Image Adapter | 0.265398 |
| LLM Prefill | 287.659972 |
| LLM Decode | 189.540453 |
| E2E（包含其它互斥成本的总时间） | 663.961221 |

逐请求视觉路径占比的平均为 **8.8924%**，Vision Tower 单独为 **8.4830%**。
该工作负载不支持“视觉编码主导端到端时延”的描述。消除这些测得视觉阶段、管理
成本为零时的理想界限约 1.098×，并非实际实现或测得的提速。模型加载、Warmup、
等价性检查、详细 Trace 均未计入 300 条 timing 汇总。

## 验证记录

- 修改前：`python -m unittest discover -s experiments/motivation_exp1/tests -v`，30/30 通过。
- 修改后：38/38 通过。新增测试覆盖关键帧限制、gap 顺序、真实间隔容差、重复
  Patch/旧 pair 恢复、无效分项统计、全局对象精确关联、Git 不可用和无效配置。
- dry-run：全部必需数据表、真实评测输入和 CUDA 检查通过。
- smoke-test：2 scenes、2786 张索引图像、36/36 图像对、79 对对象转换、
  2 个模型请求、2 条 timing、1 份详细 Trace，全部阶段通过，输出等价性 2/2。
- Smoke 使用物理 GPU 1；开始时已有其它任务占用约 5593 MiB 显存，故其时延只作
  功能检查，未替换完整实验统计，未终止其它任务。
- 最终完整命令 `--existing resume --profile-mode both` 实测返回 0；已完成的
  Profiling 正确跳过，未重新加载模型或重复 100×3 请求。
- 六组 PNG/PDF 图已生成；PNG DPI 与 PDF 文件头、真实 Trace kernel 事件已验证。
- 主结果与 Smoke 的当前源码哈希、指纹、阶段状态及原始复用文件哈希均核对通过。

机器可读证据：`outputs/motivation_exp1/verification_results.json`。
统计报告：`outputs/motivation_exp1/experiment_report.md`。

## 修改文件清单

| 文件（相对于 experiments/motivation_exp1/） | 修改目的 |
|---|---|
| `build_nuscenes_manifest.py` | 定向修正 gap 语义、关键帧限制和时间容差；保持原索引 |
| `analyze_temporal_redundancy.py` | 增加 Patch 均值、验证恢复数据完整性与当前选择 |
| `match_visual_objects.py`（新增） | 同 scene 双关键帧精确对象关联与 SSIM 条件统计 |
| `aggregate_results.py` | 增加显式 scene 组及联合证据；隔离无效阶段计时 |
| `profile_senna.py` | 仅完善汇总过滤；实际 Profiling 执行函数保持不变 |
| `plot_results.py` | 摄像头 Patch 分布、对象出现/消失、Adapter 分项及布局 |
| `generate_report.py` | 关联证据、对象统计、来源及局限说明 |
| `utils/data_utils.py` | Git 不可用的诚实来源记录、配置校验及新产物清理 |
| `audit_existing_results.py`（新增） | 可重查的旧产物审计及受保护的派生分析刷新 |
| `tests/test_manifest.py` | 配对语义、容差和配置边界测试 |
| `tests/test_aggregation.py` | 全局关联、无效计时与来源记录测试 |
| `tests/test_similarity.py` | Patch 均值及断点恢复完整性测试 |
| `config.yaml` | 明确三个目标间隔的语义注释 |
| `README.md` | 本机路径、Smoke、复用与运行指令 |
| `AUDIT.md`（新增） | 本审计、文件清单与实测交付记录 |

`run.py`、`analyze_object_persistence.py`、`utils/profiling_utils.py` 及 Senna 原始模型
推理文件、权重均未修改。没有另建实验实现目录，也没有引入合成实验数据。

## 完整实验命令

```bash
cd /data/qh/phdCode/work4/Senna && \
SENNA_MODEL_PATH=/data/qh/phdCode/work4/models/Senna \
NUSCENES_ROOT=/data/dataset/v1.0-trainval CUDA_VISIBLE_DEVICES=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/data/miniconda3/envs/Senna/bin/python -m experiments.motivation_exp1.run \
  --profile-mode both --output-dir outputs/motivation_exp1 --existing resume
```

空目录会执行完整流水线；本机已完成的阶段在来源一致时复用。重新测量时使用新目录
`outputs/motivation_exp1/fresh_run` 和空闲 GPU。后续若研究摄像头可见对象的变化，
可复用 Experiment 2 已有的实例投影，再检验全局身份变化对应的相机可见性；本实验
不据此宣称已经验证 WorldStream 的缓存复用或增量推理收益。
