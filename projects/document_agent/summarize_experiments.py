"""Archive measured live/GPU summaries and write the experiment reading guide."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent


def main():
    live = []
    for path in sorted((ROOT / 'artifacts/live').glob('20261106_*/report.json')):
        report = json.loads(path.read_text(encoding='utf-8'))
        if not report['complete']:
            raise ValueError(f'Live run incomplete: {path.parent.name}')
        live.append({'run': path.parent.name, 'report_path': str(path.relative_to(WORKSPACE)).replace('\\', '/'),
                     'report_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), **report})
    gpu_path = ROOT / 'artifacts/gpu/20261106_qwen05b_lora_v1/report.json'
    gpu = json.loads(gpu_path.read_text(encoding='utf-8'))
    if not gpu['complete']:
        raise ValueError('GPU run incomplete')
    verification = gpu_path.parent / 'checkpoint_verification.json'
    if not verification.exists():
        raise ValueError('Checkpoint reload verification must finish before delivery')
    gpu['checkpoint_verification'] = json.loads(verification.read_text(encoding='utf-8'))
    (ROOT / 'artifacts/gpu_training_report.json').write_text(json.dumps(gpu, ensure_ascii=False, indent=2), encoding='utf-8')
    (ROOT / 'artifacts/live_baseline_report.json').write_text(json.dumps(
        {'date': '2026-11-06', 'runs': live, 'total_model_requests': sum(r['summary']['model_requests'] for r in live)},
        ensure_ascii=False, indent=2), encoding='utf-8')
    curve_folder = WORKSPACE / 'docs/experiments/2026-11-06'
    curve_folder.mkdir(parents=True, exist_ok=True)
    history = gpu['sft_training']['history'] + gpu['dpo_training']['history']
    with (curve_folder / 'training_steps.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['phase', 'step', 'loss', 'grad_norm',
                                                   'elapsed_seconds', 'max_memory_allocated', 'max_memory_reserved'])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in history)
    table = []
    for run in live:
        summary = run['summary']
        table.append(f"| {run['requested_model']} | {run.get('prompt_version', 'v1')} | {run['max_tokens']} | "
                     f"{summary['e2e_pass']}/20 | {summary['provenance_accepted']}/20 | "
                     f"{summary['mean_tool_calls']:.2f} | {summary['model_requests']} |")
    training_table = []
    for stage in ('base', 'sft', 'dpo'):
        a, b = gpu[stage]['summary']['train'], gpu[stage]['summary']['holdout']
        generation = gpu[stage]['generation']
        training_table.append(f"| {stage} | {a['chosen_nll_per_token']:.6f} | {b['chosen_nll_per_token']:.6f} | "
                              f"{a['mean_margin']:.3f} | {b['mean_margin']:.3f} | "
                              f"{generation['answer_and_provenance_pass']}/{generation['total']} |")
    peaks = {s: gpu[s + '_training']['max_memory_allocated'] / 2**30 for s in ('sft', 'dpo')}
    text = f'''# 真实模型工具评测与 GPU 后训练实验

执行日期：2026-11-06（Asia/Shanghai）。本轮完成本地 API 工具调用和 RTX 4060 上的真实权重更新。

## API 工具基线

用户提供的本机网关为 `http://127.0.0.1:8788/v1`。模型 ID 来自鉴权后的 `/models`；混元请求 ID 为 `hy4-preview-f`，响应自报 `hy4-preview`。这些名称为网关标签，不是独立核验的底层模型身份。

模型仅收到 Task、通用说明和四类工具 Schema，自主选择检索、读取、审校及最终输出。金标数值、页码和 element ID 没有放进输入；检索返回真实证据是允许的工具观测。

| 请求模型 | 提示词 | max_tokens | 严格端到端通过 | 来源校验通过 | 平均工具调用 | API 请求数 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

- 严格端到端通过 = 最终 JSON 合法 + field/status/value/unit/qualifier 匹配金标 + 来源校验通过 + 引用已实际读取。字符串只忽略大小写／空白，角度单位做归一化，不用模型裁判。
- 来源校验通过不等于标准答案匹配。首轮 DeepSeek 的角度回答有“值内含单位”与金标拆字段方式不同的情况，这属于输出规范差异，不直接称为幻觉。
- 每题预算为 8 次模型响应／16 次工具执行；无自动 HTTP 重试。预算耗尽、协议错误、请求失败计入分母。
- v2 在 v1 的失败分析后明确了数值／单位拆分、证据应是承载数值的单元格、审校通过即结束。没有加入具体答案、页码或 ID。混元 v2 还增加输出 token 预算，因此不能将其变化只归因于提示词，也不能用此表作公平的模型排名。
- 两轮复用原来的 20 条事实（10 个字段 × 两种产品角色），全在 K231105。这里是开发集迭代，尚无独立来源测试或统计显著性结论。部分运行时间重叠，不用延迟作模型速度比较。

### 可讲解的真实失败

1. DeepSeek v1 `outer_diameter_subject`：读到了正确数值，但最终额外引用未读取的表头；`unobserved_evidence` 和字段／数值上下文检查拒绝了该回答。
2. DeepSeek v1 的若干题在 `audit_claim` 已接受后继续尝试不同表示，耗尽步数；停止规则是 v2 的直接改动依据。
3. 混元 v1 反复尝试表头引用、单位／限定词组合，部分输出触及长度或工具预算。完整原始响应和 `finish_reason` 留在逐题文件中。

每轮目录在 `projects/document_agent/artifacts/live/`，含 `report.json`、逐题 JSON、SQLite、代码快照。逐题 JSON 包含请求／响应、工具事件和 gold（仅用于事后评分）。API Key 不写入报告。汇总见 [live_baseline_report.json](../projects/document_agent/artifacts/live_baseline_report.json)。

## 真实权重 SFT → DPO

模型 `{gpu['model']}`，固定 revision `{gpu['revision']}`。WSL Ubuntu-22.04、RTX 4060 Laptop、BF16 基座，LoRA r=8、alpha=16、dropout=0，作用于 q/k/v/o projections，共 {gpu['lora']['trainable_parameters']:,} 个可训练参数。

数据按字段族分组：训练 16 个任务／96 对，留出 `working_length`、`light_source` 的两个产品角色，共 4 个任务／24 对。同事实派生的六类负例不会跨分组。仍是单文档、相同输出模板的微型实验。

重新使用 0.5B 模型的 tokenizer 编码，不直接复用原 7B NPZ 或小词表映射。SFT 监督 assistant 工具调用及最终答案；DPO 只监督最终回答，共同上下文和工具观测被屏蔽。

SFT：40 个 optimizer steps，梯度累积 4，学习率 2e-4。DPO：40 步，累积 2，学习率 5e-5，β=0.1；冻结训练后的 SFT 策略，在 DPO 更新前缓存其 chosen/rejected log-probs。DPO 首步 loss 约 log(2)，验证初始 policy/reference 一致。

| 阶段 | 训练 chosen NLL/token | 留出 chosen NLL/token | 训练 log-prob margin | 留出 margin | 固定证据下留出回答通过 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(training_table)}

上表 NLL 来自同一组 final-response 标签的 teacher-forcing 评测；训练曲线的 SFT loss 还监督工具调用，两者不混用。Margin 是 chosen 序列总 log-prob 减 rejected 序列总 log-prob，受长度影响；报告另存 token 均值 margin。此数据的基础模型偏好准确率已经饱和，margin 拉开不能表述为偏好准确率上涨。

**DPO 的负结果必须一起看：**留出 chosen NLL/token 从 SFT 的约 0.0023 上升到 0.1852，固定证据生成从 4/4 降至 0/4。相对偏好边距增大与 chosen 绝对概率下降同时发生，与 likelihood displacement 的现象一致；本实验不能单独确证其因果机制。DPO loss 下降不代表生成质量提升。当前保留 SFT checkpoint 作为更可用版本；DPO checkpoint 留作失败分析。下一步应独立验证更小步长／更早停止或加入 SFT 约束，同时扩充独立文档，避免继续按这 4 题调参后声称泛化。

最后一列是在给定真实工具观测后生成最终 JSON，**不是小模型自主规划的端到端 Agent 成功率**。API 工具评测和本地训练是两条实验线；本地 LoRA 不能解释云端模型通过率的变化。

训练阶段峰值 allocated 显存：SFT {peaks['sft']:.3f} GiB，DPO {peaks['dpo']:.3f} GiB。reserved 另存，二者都不等于 nvidia-smi 整卡占用。所有上下文 token 都经过 Transformer；仅在需要预测的监督位置做全词表投影以节约显存，已用小型 Qwen2 校验与完整 logits 的 loss／梯度等价。

![真实训练损失曲线](experiments/2026-11-06/training_loss.png)

原始逐步数值：[training_steps.csv](experiments/2026-11-06/training_steps.csv)。完整结果：[gpu_training_report.json](../projects/document_agent/artifacts/gpu_training_report.json)。适配器位于 `projects/document_agent/artifacts/gpu/20261106_qwen05b_lora_v1/sft_adapter` 和 `dpo_adapter`；权重不纳入 Git。

## 复现入口

API 使用 Windows Conda `ai-job-search` 的 Python。仅在本机进程中设置 `LOCAL_LLM_API_KEY`，然后执行：

```powershell
& 'C:/Users/Bingyu/666/Anaconda/envs/ai-job-search/python.exe' projects/document_agent/evaluate_live.py --list-models
& 'C:/Users/Bingyu/666/Anaconda/envs/ai-job-search/python.exe' projects/document_agent/evaluate_live.py --model deepseek-v4.1-flash --prompt-version v2
```

WSL 专用虚拟环境 `/home/bingyu/.venvs/ai2resu-gpu` 以只读方式继承已有 `physical_ex` 的 CUDA PyTorch；新增依赖安装在 venv。现有 Conda 环境未升级。启动训练时给当前进程设置 `LD_LIBRARY_PATH=/home/bingyu/miniconda3/envs/physical_ex/lib`，使用 `train_lora.py --output <新目录>`。输出目录必须不存在，以保护历史实验。

## 面试表达

Agent：我让真实模型自主调用检索、读取和审校工具，按答案与证据联合判定结果。第一次评测发现模型在审校通过后仍反复调用，也会把表头当数值证据；我据此明确输出契约和停止规则，并保留逐题轨迹比较。

后训练：我在 4060 上跑了 0.5B 的 LoRA SFT 和 DPO，用字段分组隔离训练与留出，固定 SFT 参考策略记录真实序列概率、损失和显存。能解释梯度和显存优化，也能说明为什么微型同文档实验尚不足以证明跨文档泛化。

测试口径：历史 Agent 简历 37 = 20取证 + 9协议 + 8掩码；历史完整 65 再加 6偏好 + 12损失 + 10算子。本轮新增 10项 HTTP／评分测试和 2项训练数据／梯度等价测试，完整 77项。原 65 项历史报告保留；本轮报告另存。
'''
    (WORKSPACE / 'docs/11_真实模型评测与GPU训练.md').write_text(text, encoding='utf-8')
    print('Archived live/GPU reports and experiment guide')


if __name__ == '__main__':
    main()
