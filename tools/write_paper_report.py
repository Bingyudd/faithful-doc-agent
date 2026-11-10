"""Render final measured tables without changing any experiment configuration."""
from projects.paper_common.io import ROOT,read_json


def value(x,digits=3):return '—' if x is None else f'{x:.{digits}f}'


agent=read_json(ROOT/'artifacts/paper/report_report.json')
algorithm=read_json(ROOT/'artifacts/paper/decoding_report.json')
lines=['## 已完成实验结果','',
       '作者实现对齐版本共完成2800条真实生成输出，均有独立FactKB评估记录。报告共128次运行：109次生成完成，19次超时或截断失败；109次完成运行均经评估，其中18次没有可评分的事实陈述。',
       '', '### DeepSeek 来源留出：12个资料组 × 两种模板','',
       '| 方法 | 完成／运行 | 报告平均引文合法率 | FactKB均值（有陈述报告数） | 总请求数 | 累计请求时间/s |',
       '|---|---:|---:|---:|---:|---:|']
for row in agent['summary']:
    if row['model']=='deepseek-v4.1-flash' and row['split']=='test':
        lines.append(f"| {row['method']} | {row['complete']}/{row['count']} | {value(row['citation_valid_rate'])} | {value(row['factkb_mean'])} ({row['factkb_scored']}) | {row['requests']} | {value(row['seconds'],1)} |")
lines += ['', '引文合法率是逐报告平均，且无引文时不定义；FactKB也只在有事实陈述时有数值。因此上表不同方法的简单均值差不能代替配对比较。',
          '按12个来源组聚合后，reviewed相对direct的配对FactKB差为0.160，bootstrap 95%区间[0.051, 0.290]；引文合法率差为0.035，区间[-0.006, 0.085]。这是自动评分与小规模来源组上的估计，不是人工准确率。',
          '审校版请求明显更多；修订过程中仍会引入新的机械错误，最终输出保留需人工复核状态。没有将复杂流程默认判为更优。',
          '', '### 混元稳健性子集','',
          '3个来源组、两模板、四方法，共24次运行：5次结构化完成、19次因超时或输出截断失败；5次完成结果均无事实陈述。此配置下未形成可比较的质量结果，不用于给模型排名。失败与请求成本全部保存。',
          '', '### Qwen2.5-1.5B：XSum 200条测试样本','',
          '| 方法 | ROUGE-L | FactKB | 三元组重复率 | 平均时间/s | 平均输出token | 平均峰值allocated/MiB |',
          '|---|---:|---:|---:|---:|---:|---:|']
for row in algorithm['summary']:
    if '1.5B' in row['model'] and row['split']=='test' and row['dataset']=='EdinburghNLP/xsum':
        lines.append('| '+row['method']+' | '+' | '.join(value(row[k]) for k in ['rouge_l','factkb_mean','repeated_trigram_fraction','seconds','output_tokens','peak_mib'])+' |')
lines += ['', 'context相对greedy的配对FactKB差为+0.082，95%区间[+0.033,+0.125]；ROUGE-L差为−0.023，区间[−0.031,−0.016]。来源一致性评分、参考摘要重合和推理成本存在取舍。',
          '', '### Qwen2.5-1.5B：技术文本60条测试样本','',
          '| 方法 | FactKB | 数字对应精度 | 三元组重复率 | 平均时间/s |',
          '|---|---:|---:|---:|---:|']
for row in algorithm['summary']:
    if '1.5B' in row['model'] and row['split']=='test' and row['dataset']=='fda_technical':
        lines.append('| '+row['method']+' | '+' | '.join(value(row[k]) for k in ['factkb_mean','numeric_precision','repeated_trigram_fraction','seconds'])+' |')
lines += ['', '技术文本没有人工摘要参考。静态偏置的FactKB均值较高，token变体并未带来一致收益；去语义信号的退化说明组件作用需要通过实际生成检查。上述方法差异是本轮结果，不据此承诺跨领域收益。',
          '', '### 可复现性与剩余核对','',
          '- 原始输出、失败、模型revision、配置与代码哈希均保留；自动汇总可由 `python -m tools.finalize_paper` 重建。',
          '- 数据完整性、核心公式、缓存、JSON修复、日志快照和本地界面分别测试；最终状态见 `artifacts/paper/acceptance_report.json`。',
          '- 0.5B的50条固定XSum测试样本和全部消融结果在机器可读报告中，不与1.5B的200条样本简单混算。',
          '- 固定人工队列为46项，实际人工标注状态保存在 `human_review_queue.json`。未填写的仍为pending。',
          '- 自动FactKB的512-token裁剪被逐条记录；历史HTTP请求列表存在上文说明的日志限制。',
          '- 图表、简历正文和面试材料不能把生成次数当独立任务数，不能把自动评分改称人工准确率。','']
path=ROOT/'docs/12_论文驱动项目实验报告.md'
old=path.read_text(encoding='utf-8')
marker='<!-- MEASURED_RESULTS -->'
old=old.split(marker)[0]
old=old.replace('完整结果由保存的逐条输出重新计算，汇总将写入 `artifacts/paper/decoding_report.json`、`report_report.json` 及独立 FactKB 报告。正在执行的任务和最终完成性以 `artifacts/paper/delivery_report.json` 为准；本段由交付汇总脚本更新。','')
path.write_text(old.rstrip()+'\n\n'+marker+'\n'+'\n'.join(lines),encoding='utf-8')
print('Final experiment tables written.')
