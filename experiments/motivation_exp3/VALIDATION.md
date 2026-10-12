# 实测验收（2026-10-12，Senna 环境）

完整报告：[experiment_report.md](../../outputs/motivation_exp3/experiment_report.md)。
原始执行日志和结构化验收信息保存在 `outputs/motivation_exp3/metadata/validation_logs/`
及 `metadata/validation_summary.json`。

实际执行了统一入口的 Dry Run、真实 GPU Smoke Test、默认完整实验和完整断点复验。
Dry Run 没有创建结果目录；完整运行 11 个阶段均完成，复验 11 个阶段的输出哈希均通过，
没有新增图像前向。当前 GPU 为 RTX 3090，原始 loader 精度为 FP16。

| 验证 | 实际结果 |
| --- | --- |
| E3 单元 + 真实输出回归 | 35 项全部通过，无跳过 |
| E1 常规回归 | 30 项全部通过 |
| E2 常规回归 | 40 项通过，2 项真实数据测试单独执行 |
| E2 官方投影真实数据回归 | 2 项全部通过 |
| 原始 / 插桩视觉输出 | 3 批通过；tower 和压缩输出最大绝对误差均为 0 |
| FP16 统计、坐标和物理 ID | 数值、实际 processor 标记像素、真实关联检查通过 |
| Scene bootstrap | 与 E1 原实现的真实样本均值和 CI 在 1e-12 内一致 |
| 编译和格式 | compileall、git diff --check 通过 |

Smoke Test 使用 2 scenes、100 物体对、150 张独立图像，生成六类图和报告。
完整实验使用 10 scenes、六路相机、3000 物体对、709 个不同物理实例，完整编码 1706 张
独立图像，无失败图像；主面积加权分析有 2011 个有效同实例对。

运行中发现大型 CSV 的时间标签可能被 pandas 分块推断为混合类型，已统一为字符串并加入
数字标签/混合标签回归用例。旧运行已归档；当前完整结果由修正版重新完整执行产生。

Senna 所选 hidden-state 23 的同实例 cosine similarity 均值 0.75453、中位 cosine distance
0.20816、中位 relative L2 0.66295；重复输入对象描述符最大 cosine distance 5.96e-08。
同类别匹配负例有 921 对，匹配 ΔS 均值 0.06841，场景等权 ΔS CI95
[0.04976, 0.08516]；两组分布 overlap 为 0.76764。
这些结果支持被测 bbox-pooled 中间表示跨帧并非不变，也显示身份相似度分布有较大重叠。
不能据此推出驾驶任务精度、缓存正确性、复用阈值或 WorldStream 方法有效性。

完整产物为 20 个 CSV、六类 PNG、六类 PDF、30 个真实案例可视化、1706 个离线描述符
分片、JSON 统计和实验报告。本地要求的阶段均实际跑通；S2 / anyres 空间映射不属于当前
本地 checkpoint，其配置会被明确拒绝。

```bash
/home/qh/miniconda3/envs/Senna/bin/python experiments/motivation_exp3/run_exp3.py --config experiments/motivation_exp3/config.yaml --run-all
SENNA_EXP3_OUTPUT=outputs/motivation_exp3 /home/qh/miniconda3/envs/Senna/bin/python -m unittest discover -s experiments/motivation_exp3/tests -v
```
