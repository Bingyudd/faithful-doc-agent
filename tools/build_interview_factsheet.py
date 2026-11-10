"""Build the interview fact-boundary sheet from the resume ledger and metric audit.

The sheet answers one question per resume number: can it be recomputed in front of
an interviewer, can it only be quoted from a stored summary, or is the detail not
recorded at all and should not be claimed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from projects.paper_common.io import ROOT, read_json

# metric path prefix -> (producing artefact, tracked in git?, strength, note)
SOURCES = {
    'dataset': ('data/paper/provenance.json、data/paper/sources.json', True, 'A',
                '冻结清单与计数，仓库内可直接读'),
    'models': ('data/paper/provenance.json', True, 'A', '固定 revision 记录'),
    'code_limits': ('configs/paper_frozen.json', True, 'A', '冻结配置常量'),
    'comparison': ('artifacts/paper/decoding_report.json', True, 'A', '含逐样本重复率与 EOS 率'),
    'decoding_selected': ('artifacts/paper/decoding_report.json', True, 'A', '含逐样本明细与汇总'),
    'agent': ('artifacts/paper/report_report.json + artifacts/paper/raw/reports_v3/**（未入库）', False, 'B',
              '汇总在库；逐份报告与请求明细只在本地 raw'),
    'revision': ('artifacts/paper/raw/reports_v3/**（未入库）', False, 'B',
                 '修订轮次与告警明细来自本地 raw，未提交'),
    'cross_strategy': ('由 agent.direct 与 agent.reviewed 两个计数相除', True, 'A',
                       '两个整数当场可算：(19-9)/19'),
    'training': ('projects/document_agent/artifacts/gpu_training_report.json', True, 'A',
                 '计数与损失曲线在库；适配器与逐步日志未入库'),
}

STATIC_HEAD = """# 面试事实边界与自测清单

由 `tools/build_interview_factsheet.py` 从 `resumes/paper_claims.json` 与
`artifacts/paper/resume_metric_audit.json` 生成，禁止手改。

用法：面试前只看第二、三节。**A 级**数字可以当场复算；**B 级**只能引用汇总并说明"逐样本明细未随仓库提交"；
**C 级**是记录里没有的东西，背熟了也会被追问拆穿，一律不讲。

"""

STATIC_TAIL = """
## 四、C 级：记录里没有，不要在面试里讲

| 别讲的说法 | 为什么不能讲 | 如果被问到怎么答 |
|---|---|---|
| "打印了每步解码的中间张量分布，定位到二次 softmax 与 cos 负值截断" | 日志里没有张量 dump 记录；能讲的是代码与消融对照的差异 | "现象是 3-gram 复读率 65.94%，我对照了三种偏置实现和去语义消融（74.43%），差异指向来源信号未归一化；具体张量没有留档，这里只能讲推理" |
| "单测容差 1e-5" | 单测里是断言等价，没有记录这个具体阈值 | "单测比较增量与全序列 logits／注意力切片是否一致，用的是默认断言容差，没有专门记 1e-5" |
| "审校侧 4 份因证据不足被 missing 拦截" | 只有 factkb_n=20 这个事实，成因未逐份归档 | "FactKB 只对有事实陈述的报告计分：直接生成 23 份、审校 20 份；具体少掉的 4 份原因没有逐份归档" |
| "用 embedding／LangGraph／AutoGen 做过对比" | 完全没有做过 | "没有做；我选 BM25＋来源隔离是因为编号、条款号和单位必须精确匹配，这是我当时的取舍，不是对比结论" |
| "多卡／分布式训练、7B 以上模型" | 只有单卡 4060 与 0.5B／1.5B 推理 | "单卡 8GB，最大只到 1.5B 推理与 0.5B LoRA；多卡和大模型没有条件做" |
| "QPS／并发／吞吐优化" | 没有压测记录 | "没有做压测；能讲的是单条时延与峰值显存这种单请求指标" |
| "上线／真实用户／返工率下降" | 实习期间没有可引用的业务埋点 | "实习是内部资料准备，没有做业务度量；我能讲的是把多源材料组织成初稿的流程与后来抽象出的核验需求" |
| "结论有统计显著性／多种子重复" | 每档只跑一次，样本 24／50／60／200 条 | "都是单次运行的小样本探索，用来判断方向，不做显著性结论" |
| "零交叉污染／100% 可复现／零幻觉" | 工程上没有零 | "来源隔离降低交叉风险；可复现指的是固定 revision、配置哈希与冻结清单" |

## 五、自测问题（能答再上场）

标记：**A** 可当场复算或读代码回答；**B** 只能引汇总；**C** 不要答，按第四节口径转述。

### Agent 项目

1. （A）20 组来源、375 块、8／12 怎么来的？→ `data/paper/provenance.json`＋`data/paper/source_quality_report.json`
2. （A）为什么按申报方隔离而不是按行随机？→ 同一份申报资料内部字段互斥，随机拆会泄漏
3. （A）逻辑预算 20 次和 HTTP 重试为什么要分开？→ `projects/report_agent/engine.py` 两处计数不共享
4. （B）24 份里初稿 12 条、终稿 9 条怎么数的？→ 汇总在库，逐份在本地 raw
5. （B）剩余 9 条为什么没消？→ 只能答机制（2 轮上限、子串必须连续、宁可留告警交人工）
6. （A）19 条与 12 条是什么关系？→ 前者是"直接生成 vs 审校"跨策略；后者是"审校内部初稿 vs 终稿"
7. （A）52.6% 怎么算？→ (19-9)/19，当场算
8. （B）FactKB 是谁的分数、怎么算？→ 外部评分模型对事实陈述打分；本仓库只存分数不存模型权重
9. （A）成本为什么涨到 9.25 次/报告？→ 分解＋审校多轮；`agent.reviewed.requests_total/24`
10. （C）有没有做 embedding 检索对比？→ 见第四节
11. （C）并发、缓存、可观测性？→ 没有做，不要编
12. （A）配置哈希恢复章节怎么保证幂等？→ 三元组哈希＋状态文件校验，代码在 engine.py
13. （B）24 份样本够不够？→ 小样本探索，方向验证；扩样设计可以讲（按申报方分层）

### 算法项目

14. （A）65.94%→2.23% 是什么口径？→ 50 条开发集重复 3-gram 占比，`comparison.xsum_dev_token`
15. （A）74.43% 的消融说明了什么？→ 去掉语义项后重复飙升，说明来源信号是主因
16. （A）为什么 FP32 算 JSD？→ 概率下溢；代码 `projects/faithful_decoding/core.py`
17. （A）logaddexp 怎么得到 log M？→ log M = logaddexp(log P, log Q) − log 2
18. （A）O(S²d)→O(Sd) 的推导？→ 归一化后均值向量的点积与成对点积期望相等
19. （A）双流 KV Cache 怎么同步？→ 同一 y_t 追加到两条分支；单测在 test_core.py
20. （A）峰值 2.95 GiB 怎么测的？→ `decoding_selected.main_xsum_token.mean_peak_gib`，逐条记录取均值
21. （B）ROUGE-L 为什么降？→ 强制贴合来源会改变表面重合；能讲机制，不给因果证明
22. （A）FactKB 0.694→0.776 的样本量？→ 200 条测试
23. （C）1.5B 结论能否迁到 7B？→ 没有做，不要声称
24. （A）三种偏置实现的差异？→ 代码里三种策略分支＋消融

### 后训练

25. （A）120 对／86 对怎么构造？→ `preference_data.py`；同任务同观测，只换最终结论与引用
26. （A）96／24 怎么分？→ 按字段家族，working_length＋light_source 整体留出
27. （A）Loss Masking 掩的是哪些位置？→ Prompt 与工具观测置 -100，只监督 assistant 终答
28. （A）参考策略校正是什么？→ 冻结 SFT 策略的 chosen/rejected log-prob 缓存
29. （A）SFT 4/4、DPO 0/4 怎么测？→ 固定黄金工具观测下的 4 个留出任务，规则化答案＋来源校验
30. （B）DPO 为什么退化？→ 能讲证据（边距升、chosen NLL 升、生成掉）与已记录的解释假设；正式因果实验另属一条分支，不在本简历证据内
31. （A）SimPO 与 DPO 的差别？→ 去 reference、长度归一化、目标裕度 γ
32. （C）"手写实现并完成梯度等价单测"能不能现场写？→ 能讲公式与代码位置；没练过就不要说"现场推导"

## 六、复算命令

```powershell
# 账本重算（不训练、不调用模型）
python -m tools.resume_metric_audit
# 逐句账本 + PDF 文本一致性
python -m tools.verify_resume_reconstruction --pdf
# 指标单测
python -m unittest tools.test_resume_metrics
```
"""


def strength(path):
    return SOURCES.get(path.split('.')[0], ('未知来源', False, 'C', ''))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'docs/17_面试事实边界与自测清单.md')
    args = parser.parse_args()
    claims = read_json(ROOT / 'resumes/paper_claims.json')
    audit = read_json(ROOT / claims['metric_audit'])
    lines = [STATIC_HEAD, '## 一、简历数字溯源表\n',
             '| 版本 | 环节 | 简历表述 | 指标路径 | 现值 | 级 | 来源 |',
             '|---|---|---|---|---|---|---|']
    seen = set()
    for entry in claims['entries']:
        for placeholder, spec in entry['metrics'].items():
            key = spec['path']
            if key in seen:
                continue
            seen.add(key)
            value = audit
            for part in key.split('.'):
                value = value[part]
            shown = format(value, spec['format'])
            source, tracked, level, _ = strength(key)
            lines.append(f"| {entry['version']} | {entry['label']} | {entry['text'][:38]}… | `{key}` | {shown} | {level} | {source} |")
    lines.append('\n## 二、级别定义\n')
    lines.append('| 级 | 含义 | 面试时怎么用 |\n|---|---|---|')
    lines.append('| A | 数据或配置在仓库内，可当场读文件、算除法 | 直接答，必要时打开文件 |')
    lines.append('| B | 汇总在库，逐样本明细只在本地 raw（未提交） | 引用汇总，说明明细未随仓库提交 |')
    lines.append('| C | 没有记录 | 按第四节口径转述，不编细节 |')
    lines.append('\n## 三、B 级数字的说明模板\n')
    lines.append('> "这个数来自留出运行的汇总，我把逐份结果和请求日志留在本地，仓库里提交的是汇总与复算脚本；'
                 '如果要复现，需要在同一台机器上用同一批来源重跑，成本大约 X。"（把 X 换成实际时长，不要编）')
    lines.append(STATIC_TAIL)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text('\n'.join(lines), encoding='utf-8')
    print('wrote', args.out, 'rows', len(seen))


if __name__ == '__main__':
    main()
