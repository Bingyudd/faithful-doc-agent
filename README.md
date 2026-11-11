# 证据可核验的长文档生成：受控 Agent、忠实解码与后训练验证

两个独立项目，共用一份可审计的评测口径：**先把"证据是否可核验"做成可测量对象，再让生成、干预和训练都围绕这个对象做对照。**

- **项目一｜受控文档 Agent**：FDA 510(k) 技术资料的结构化生成与审校。来源隔离检索 → 模板化生成 → 机械核验 → 最多 2 轮局部修订 → Diff 人工复核。
- **项目二｜上下文忠实解码（CAD）**：不改权重，用来源区间的注意力与语义相似度对 logits 做偏置，压低先验幻觉；含三种偏置实现、双流 KV Cache 与消融。
- **后训练验证**：在 0.5B 上完成 LoRA SFT／DPO 留出验证，构造同答案、证据无效的强负例偏好数据，SimPO 只做损失与梯度核验。

结果总表见 [RESULTS.md](RESULTS.md)（由脚本从指标账本重算，不手写数字）。

## 关键结果（同源小样本，方向验证而非统计结论）

| 维度 | 结果 |
|---|---|
| 审核可靠性 | 24 份留出报告：总机械告警 19 → 9（净降 52.6%，相对直接生成）；零告警报告 15/24 → 20/24；逐引文通过 141/152 → 154/159 |
| 事实评分 | FactKB 0.708 → 0.842（仅对含事实陈述的报告计分：23 → 20 份） |
| 成本 | 平均请求 1.04 → 9.25 次/报告：明确是"用请求换可靠性"，未做缓存与模型路由优化 |
| 解码 | XSum 200 条：Context FactKB 0.694 → 0.776；逐字 ROUGE-L 0.172 → 0.148（预期内取舍）；单条 1.68 → 2.34 s |
| 复读退化 | 开发集 50 条重复 3-gram 65.94% → 2.23%；去语义消融回升至 74.43% |
| 后训练 | 固定黄金工具观测下 base 0/4 → SFT 4/4 → DPO 0/4；DPO 的负结果原样保留 |

## 快速验证（不需要任何 API Key）

```bash
python -m tools.resume_metric_audit              # 从留存报告重算全部指标
python -m tools.verify_resume_reconstruction     # 逐句账本与实验数值一致性
python -m unittest tools.test_resume_metrics     # 指标口径单测
python -m unittest projects.paper_common.test_integrity projects.paper_common.test_provider \
                   projects.report_agent.test_engine projects.report_agent.test_ui
```

需要模型与 GPU 的部分（真实 API 调用、LoRA 训练、CAD 生成）无法脱机复现，依赖本机的 API 网关与 WSL CUDA 环境，见 [REPRODUCE.md](REPRODUCE.md)。

## 仓库结构

```
projects/report_agent/     受控 Agent：工具运行时、状态机、核验与审校入口
projects/faithful_decoding/ CAD：三种 logits 偏置、JSD、双流 KV Cache 与单测
projects/paper_common/      数据准备、事实评分、来源质量、Provider 快照
configs/                    冻结的生成配置与公式版本
artifacts/paper/            汇总报告（含逐样本明细的解码报告）
tools/                      指标账本、交付校验、复现脚本
```

## 设计与取舍文档

- [ARCHITECTURE.md](ARCHITECTURE.md)：数据流、状态机与双流解码的图与边界
- [FAILURES.md](FAILURES.md)：四个真实失败的复现、定位与未解决项
- [REPRODUCE.md](REPRODUCE.md)：哪些能脱机重算、哪些需要本机环境

## 边界声明

- 全部实验为**同源小样本**（24 份报告、50／200／60 条解码样本），每档单次运行，没有多种子与显著性检验。
- FactKB 是外部自动评分，不是人工判定；机械告警是规则命中，误报／漏报未标注。
- 生成指标使用固定黄金工具观测，不代表自主工具调用的端到端成功率。
- 来源为公开的 FDA 510(k) 文档（3 份），未包含任何公司内部资料；本仓库不附带 PDF 原件。

## 背景

任务背景来自医疗器械注册资料整理的实际流程：沙特以胆道镜注册为主、美国以临床检测报告为主、巴西是较完整的注册文件（作参考口径）；覆盖胆道镜、膀胱镜、主机以及导丝、输尿管导引鞘等产品，单份初稿的生成耗时约 1—3 小时。
本仓库只保留可以公开复算的部分（公开 FDA 文档、代码、汇总指标），不包含任何公司内部资料、真实产品数据或原始生成日志。

