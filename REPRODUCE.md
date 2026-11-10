# 复现指南

分三层：**能脱机重算的**、**需要本机环境才能重跑生成的**、**无法复现的**。

## 一、脱机可重算（不需要任何 API Key 或 GPU）

```bash
# 环境：Python 3.11 + pydantic / pdfplumber / numpy / pypdf / reportlab
python -m tools.resume_metric_audit        # 从留存报告重算 53 项指标 -> artifacts/paper/resume_metric_audit.json
python -m tools.verify_resume_reconstruction   # 逐句账本与实际数值一致性
python -m unittest tools.test_resume_metrics
python -m unittest projects.paper_common.test_integrity \
                   projects.paper_common.test_provider \
                   projects.report_agent.test_engine \
                   projects.report_agent.test_ui
```

预期：账本重算结果与仓库内存储的 `resume_metric_audit.json` 完全一致（脚本会断言），测试全绿。
数字来源映射见 `tools/build_interview_factsheet.py` 里的 A/B/C 分级。

## 二、需要本机环境才能重跑

| 环节 | 依赖 | 入口 |
|---|---|---|
| 真实 API 生成与审校（24 份 × 4 策略） | DeepSeek／混元网关与 key（不入库） | `python -m tools.run_report_campaign` |
| CAD 生成（1.5B／0.5B） | WSL + CUDA + 固定 revision 权重 | `python -m projects.faithful_decoding.cli` |
| LoRA SFT／DPO | WSL CUDA + PEFT + 0.5B 权重 | `python projects/document_agent/train_lora.py --output <新目录>` |
| 本地审校界面 | streamlit | `python -m streamlit run projects/report_agent/app.py` |

原始请求日志、适配器权重、PDF 原件都在 `.gitignore` 里，没有随仓库分发：
没有它们，第二节只能重跑，不能"验证历史"。

## 三、无法复现的部分

| 内容 | 原因 |
|---|---|
| 逐份报告的原始 response 与 token 级日志 | 未入库（体积 + 可能含业务文本） |
| LoRA 适配器与训练日志 | 未入库；只有汇总指标与损失曲线 |
| FactKB 评分模型权重 | 外部模型，只存分数 |
| 人工复核 46 项标注 | 仍为 pending，需要人工判断 |

## 四、目录与产物对应

| 产物 | 路径 | 说明 |
|---|---|---|
| 指标账本 | `artifacts/paper/resume_metric_audit.json` | 全部数字的单一来源 |
| 审校汇总 | `artifacts/paper/report_report.json`、`report_campaign_report.json` | 四策略 × 24 份 |
| 解码明细 | `artifacts/paper/decoding_report.json` | 含逐样本重复率、FactKB、显存 |
| 后训练汇总 | `projects/document_agent/artifacts/gpu_training_report.json` | 训练曲线、逐对 log-prob、生成结果 |
| 示例报告 | `artifacts/paper/examples/reviewed_report.{md,json}` | 可直接阅读的终稿样例 |

## 五、已知环境坑

- Windows 下 `tempfile` 以 0700 创建的目录在受限令牌环境中会被拒写，会让两项测试出现假错误；普通终端无此问题。
- `*.json`、`*.jsonl` 由脚本以 CRLF 写出，索引是 LF，`git status` 可能显示 M 而 `git diff` 为空；用 `git add --renormalize` 或 `git checkout --` 处理，不要当成内容变更。
- WSL 路径含空格或引号时用 `wsl --exec bash -lc '...'`，并给训练进程设 `LD_LIBRARY_PATH` 指向已有 CUDA 库目录。
