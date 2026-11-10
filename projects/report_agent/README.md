# 技术文档生成与审校

本项目把模板化报告生成迁移到公开器械资料。输入是资料、产品范围和目标模板，输出是可审校的章节、原文引用、缺失项与修订记录。所有输出是技术初稿。

## 运行

从本 worktree 根目录使用 Windows Conda `ai2resu-paper`：

```powershell
conda run -n ai2resu-paper python -m projects.report_agent.cli generate --split dev --limit 1 --methods reviewed --templates overview
conda run -n ai2resu-paper python -m streamlit run projects/report_agent/app.py --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
conda run -n ai2resu-paper python -m unittest projects.report_agent.test_engine projects.report_agent.test_ui -v
```

优先读取 `LOCAL_LLM_API_KEY`；否则从用户授权的 WSL 网关配置读取唯一启用的应用 key。密钥不进入请求日志。默认地址 `http://127.0.0.1:8788/v1`。

## 四种方法

| 参数 | 行为 |
|---|---|
| `direct` | 合并各模板章节检索出的证据，一次生成全部章节 |
| `agentic` | 全模板控制器调用 `task_decompose` 与 `retrieve_generate`，自主选择查询 |
| `decomposed` | 固定章节问题，分别检索和生成，再确定性合并 |
| `reviewed` | 在章节生成上增加机械检查、模型批注与最多两轮局部修订 |

AgenticIR 和 DecomposedIR 是 SIGIR 2025 论文方法的迁移实现。原文使用金融与气候资料、AutoGen 和不同模型；本实现使用本机网关、BM25、类型化输出和公开 FDA 文档，不能拿结果直接对应原论文表格。复用原项目的 HTTP 传输、文本规范化和 BM25 公式，未复刻其原有模型训练。

## 输出与恢复

每个运行目录含 `result.json`、`state.json`、`http.json`，失败另存 `failure.json`。身份包含源文件哈希、模板、方法、模型、代码和预算。已完成运行直接读取缓存；章节流程在已完成章节后继续。控制器在中断后重新规划未完成任务，保留累计预算。

交付的SnapshotProvider为新请求生成不可变消息快照。冻结实验的历史HTTP request列表存在共享引用限制，不能用于逐字网络回放；原始response、业务事件和结果保留。详见实验报告“日志审计补充”。

`result.json` 中 `report` 为 Pydantic 校验后的内容，`evidence` 是实际检索观测，`audits` 是机械检查，`revisions` 保留修订前后。英文原文引用保持原样；界面提供中文操作提示。

## 评测边界

20 个申报方家族，8 开发／12 测试，每组两类模板。按同一文档家族隔离，不使用旧项目的 20 题作为测试。测试前运行 `python -m tools.freeze_paper`，之后修改核心配置会拒绝测试运行。

机械检查只验证引用存在、引文原样、目标文档和数字单位对应；不会证明完整语义蕴含。章节覆盖率仅指模板章节齐全。独立 FactKB 与固定人工复核队列用于进一步检查，模型批注与人工复核状态分开记录。

论文与固定提交见 `data/paper/provenance.json`。不包含在线服务部署、OCR、完整监管申报或合规判断。
