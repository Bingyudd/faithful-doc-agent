"""Recompute resume-sized results, with denominators, from saved experiments."""
from collections import Counter,defaultdict
import json
import statistics
from projects.paper_common.io import ROOT,read_json,write_json,digest

METHODS=['direct','agentic','decomposed','reviewed']
STATES=['supported','missing','conflict','review_required']


def build_audit():
    report=read_json(ROOT/'artifacts/paper/report_report.json')
    decoding=read_json(ROOT/'artifacts/paper/decoding_report.json')
    original=read_json(ROOT/'artifacts/paper/paper_equations_dev_report.json')
    sources=read_json(ROOT/'data/paper/sources.json')
    corpus=read_json(ROOT/'data/paper/corpus.json')
    audit={'schema':1,'dataset':{'source_groups':len(sources['sources']),'blocks':len(corpus['elements']),
           'dev_groups':sum(d['split']=='dev' for d in sources['sources']),
           'test_groups':sum(d['split']=='test' for d in sources['sources']),
           'heldout_reports':sum(r['model']=='deepseek-v4.1-flash' and r['split']=='test' for r in report['samples']),
           'templates':2,'methods':len(METHODS),'generated_outputs':len(decoding['samples'])},
           'models':read_json(ROOT/'data/paper/provenance.json'),'agent':{},'decoding':decoding['summary'],
           'code_limits':read_json(ROOT/'configs/paper_frozen.json')['report']}
    input_paths=['artifacts/paper/report_report.json','artifacts/paper/decoding_report.json',
                 'artifacts/paper/paper_equations_dev_report.json','data/paper/sources.json','data/paper/corpus.json',
                 'projects/document_agent/artifacts/gpu_training_report.json','projects/document_agent/artifacts/preference_report.json']
    raw_runs=[]
    for method in METHODS:
        rows=[r for r in report['samples'] if r['model']=='deepseek-v4.1-flash' and r['split']=='test' and r['method']==method]
        assert len(rows)==24 and all(r['status']=='complete' for r in rows)
        summary=next(s for s in report['summary'] if s['model']=='deepseek-v4.1-flash' and s['split']=='test' and s['method']==method)
        states=Counter();errors=Counter();bad=0
        for row in rows:
            raw=read_json(ROOT/row['path']);raw_runs.append((row,raw))
            states.update(s['status'] for s in raw['report']['sections'])
            found=[e for es in raw['audits'].values() for e in es]
            errors.update(e['code'] for e in found);bad+=bool(found)
            input_paths.append(row['path'])
        valid=sum(r['valid_citations'] for r in rows);total=sum(r['citations'] for r in rows)
        audit['agent'][method]={'model':'deepseek-v4.1-flash','split':'test','n':len(rows),'source_groups':12,'sections':72,'section_coverage':summary['section_coverage'],
            'citation_macro':summary['citation_valid_rate'],'citation_macro_n':sum(r['citation_valid_rate'] is not None for r in rows),
            'citation_pooled':{'passed':valid,'total':total,'rate':valid/total},
            'no_error_reports':24-bad,'mechanical_error_records':sum(errors.values()),'error_types':dict(errors),
            'factkb':summary['factkb_mean'],'factkb_n':summary['factkb_scored'],
            'requests_total':summary['requests'],'requests_mean':summary['requests']/24,
            'prompt_tokens_mean':summary['prompt_tokens']/24,'completion_tokens_mean':summary['completion_tokens']/24,
            'tokens_mean':(summary['prompt_tokens']+summary['completion_tokens'])/24,
            'request_seconds_mean':summary['seconds']/24,
            'states':{s:states[s] for s in STATES},'state_fractions':{s:states[s]/72 for s in STATES}}
    counts=Counter();before_types=Counter();after_types=Counter();initial_states=Counter();section_records=[]
    for row,raw in raw_runs:
        if row['method']!='reviewed':continue
        revisions=defaultdict(list)
        for rev in raw['revisions']:
            revisions[rev['section']].append(rev)
            before=len(rev['before_errors']);after=len(rev['after_errors'])
            counts['rounds']+=1;counts['round_before']+=before;counts['round_after']+=after
            counts['count_decreases']+=max(0,before-after);counts['count_increases']+=max(0,after-before)
            bc=Counter(json.dumps(e,sort_keys=True) for e in rev['before_errors']);ac=Counter(json.dumps(e,sort_keys=True) for e in rev['after_errors'])
            counts['removed_fingerprints']+=sum((bc-ac).values());counts['added_fingerprints']+=sum((ac-bc).values())
        for section in raw['report']['sections']:
            revs=revisions.get(section['id'],[])
            before=revs[0]['before_errors'] if revs else raw['audits'][section['id']]
            after=raw['audits'][section['id']]
            counts['sections']+=1;counts['revised_sections']+=bool(revs)
            counts['initial_errors']+=len(before);counts['final_errors']+=len(after)
            counts['initial_error_sections']+=bool(before);counts['final_error_sections']+=bool(after)
            counts['initial_bad_sections_cleared']+=bool(before) and not after
            counts['initial_clean_sections_became_bad']+=not before and bool(after)
            before_types.update(e['code'] for e in before);after_types.update(e['code'] for e in after)
            initial_states[(revs[0]['before'] if revs else section)['status']]+=1
            section_records.append({'run_id':raw['run_id'],'section':section['id'],'rounds':len(revs),
                                    'before':len(before),'after':len(after),'path':row['path']})
    audit['revision']={**dict(counts),'net_reduction_fraction':(counts['initial_errors']-counts['final_errors'])/counts['initial_errors'],
        'before_types':dict(before_types),'after_types':dict(after_types),'initial_states':{s:initial_states[s] for s in STATES},
        'sections_detail':section_records,'interpretation':'mechanical issue records, not human-labeled true errors; round sums include repeated checks; fingerprints may move with statement reordering',
        'verifier_precision_recall':'unavailable: no human-labeled true-positive/false-negative denominator'}
    direct,reviewed=audit['agent']['direct'],audit['agent']['reviewed']
    audit['cross_strategy']={'direct_errors':direct['mechanical_error_records'],'reviewed_errors':reviewed['mechanical_error_records'],
        'error_reduction_fraction':(direct['mechanical_error_records']-reviewed['mechanical_error_records'])/direct['mechanical_error_records'],
        'perfect_reports_direct':direct['no_error_reports'],'perfect_reports_reviewed':reviewed['no_error_reports'],
        'interpretation':'same 24 held-out reports, direct generation versus the review pipeline; distinct from the within-pipeline initial-to-final revision count'}
    audit['comparison']={}
    for prior in original['summary']:
        current=next(s for s in decoding['summary'] if all(s[k]==prior[k] for k in ['model','dataset','split','method']))
        old_rows={r['id']:r for r in original['samples'] if all(r[k]==prior[k] for k in ['model','dataset','split','method'])}
        new_rows={r['id']:r for r in decoding['samples'] if all(r[k]==prior[k] for k in ['model','dataset','split','method'])}
        assert old_rows.keys()==new_rows.keys()
        # IDs alone are insufficient: require the same full sample hash, task, model and generation budget.
        for sample_id in old_rows:
            old_raw=read_json(ROOT/old_rows[sample_id]['path']);new_raw=read_json(ROOT/new_rows[sample_id]['path'])
            for key in ['sample_hash','task','model']:assert old_raw['identity'][key]==new_raw['identity'][key]
            for key in ['max_source_tokens','max_new_tokens','delta','delta_min','delta_max','attention_weight','cad_alpha']:
                assert old_raw['identity']['config'][key]==new_raw['identity']['config'][key]
        key=('xsum' if prior['dataset']=='EdinburghNLP/xsum' else 'technical')+'_dev_'+prior['method']
        audit['comparison'][key]={'model':prior['model'],'dataset':prior['dataset'],'split':prior['split'],'method':prior['method'],
             'n':prior['count'],'before_rep':prior['repeated_trigram_fraction'],
             'after_rep':current['repeated_trigram_fraction'],'before_eos':prior['eos_rate'],'after_eos':current['eos_rate'],
             'sample_ids':sorted(old_rows),'matched_inputs_and_budgets':True}
    audit['decoding_selected']={}
    for summary in decoding['summary']:
        rows=[r for r in decoding['samples'] if all(r[k]==summary[k] for k in ['model','dataset','split','method'])]
        for metric in ['rouge_l','seconds','output_tokens','peak_mib','repeated_trigram_fraction','factkb']:
            values=[r[metric] for r in rows if r.get(metric) is not None]
            target=summary.get('factkb_mean' if metric=='factkb' else metric)
            assert (not values and target is None) or abs(statistics.mean(values)-target)<1e-9
        summary['max_observed_peak_mib']=max(r['peak_mib'] for r in rows)
        summary['mean_peak_gib']=summary['peak_mib']/1024
        if summary['split']=='test':
            key=('small_' if '0.5B' in summary['model'] else 'main_')+('xsum_' if summary['dataset']=='EdinburghNLP/xsum' else 'technical_')+summary['method']
            audit['decoding_selected'][key]=summary
    g=read_json(ROOT/'projects/document_agent/artifacts/gpu_training_report.json');p=read_json(ROOT/'projects/document_agent/artifacts/preference_report.json')
    audit['training']={'model':g['model'],'sft_steps':g['sft_training']['steps'],'dpo_steps':g['dpo_training']['steps'],
       'trainable_parameters':g['lora']['trainable_parameters'],'preference_pairs':p['preference_pairs'],
       'same_answer_invalid_evidence_pairs':p['same_answer_invalid_evidence_pairs'],'train_pairs':g['split']['train_pairs'],
       'holdout_pairs':g['split']['holdout_pairs'],
       'generation_passes':{k:{'passed':g[k]['generation']['answer_and_provenance_pass'],'total':g[k]['generation']['total']} for k in ['base','sft','dpo']},
       'simpo_scope':'tensor/loss/gradient validation only; no SimPO model training',
       'scope':'historical one-source pilot; fixed gold tool observation, four held-out tasks, not new paper-project training'}
    audit['source_hashes']={p:digest((ROOT/p).read_bytes()) for p in sorted(set(input_paths))}
    return audit


def render(a):
    lines=['# 简历重构：真实指标核对清单','','本清单由保存的汇总与原始运行重新计算，不重新训练或调用模型。',
      '模型：Qwen2.5-1.5B-Instruct／Qwen2.5-0.5B-Instruct。方法别名：agentic_plan→agentic，section_decompose→decomposed，audit→reviewed。',
      '', '## Agent：主模型96次来源留出','','12来源×2模板×4策略。每策略24报告、72章节。章节覆盖是结构指标；FactKB和引文宏平均排除无事实／无引文报告。',
      '', '| 方法 | 章节覆盖 | 引文宏平均(n) | 逐引文通过 | 无机械问题报告 | 机械问题条数 | FactKB(n) | 请求/报告 | prompt token/报告 | completion token/报告 | 总token/报告 | 请求耗时s/报告 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in METHODS:
        r=a['agent'][m];c=r['citation_pooled']
        lines.append(f"| {m} | {r['section_coverage']:.1%} | {r['citation_macro']:.2%} ({r['citation_macro_n']}) | {c['passed']}/{c['total']}={c['rate']:.2%} | {r['no_error_reports']}/24 | {r['mechanical_error_records']} | {r['factkb']:.4f} ({r['factkb_n']}) | {r['requests_mean']:.2f} | {r['prompt_tokens_mean']:.2f} | {r['completion_tokens_mean']:.2f} | {r['tokens_mean']:.2f} | {r['request_seconds_mean']:.2f} |")
    lines+=['','平均耗时为累计HTTP尝试耗时除以报告数，不等于并发批次墙钟时间；token采用接口报告口径。','',
             '| 方法 | supported | missing | conflict | review_required |','|---|---:|---:|---:|---:|']
    for m in METHODS:
        r=a['agent'][m];lines.append('| '+m+' | '+' | '.join(f"{r['states'][s]}/72 ({r['state_fractions'][s]:.2%})" for s in STATES)+' |')
    r=a['revision']
    lines+=['','## 修订前后：证据审校24报告／72章节','',
      f"- {r['revised_sections']}章节发生修订，共{r['rounds']}轮；每章节最多2轮。",
      f"- 每节首次初稿→最终稿：机械问题记录{r['initial_errors']}→{r['final_errors']}，净减少{r['net_reduction_fraction']:.1%}；有问题章节{r['initial_error_sections']}→{r['final_error_sections']}。",
      f"- 初始有问题的{r['initial_error_sections']}节中，{r['initial_bad_sections_cleared']}节最终清零；另有{r['initial_clean_sections_became_bad']}个原本无机械问题的章节最终出现问题。",
      f"- 逐轮累加：检查前{r['round_before']}条、检查后{r['round_after']}条；计数下降量{r['count_decreases']}、上升量{r['count_increases']}。同一问题可能被多轮重复检查。",
      f"- 错误记录指纹差分：移除{r['removed_fingerprints']}、新增{r['added_fingerprints']}；受陈述位置变化影响，不等于独立真实错误数。",
      '- 没有人工真值分母，不能给出机械校验器的真实precision／recall或“幻觉拦截准确率”。',
      '- 初始错误类型：'+json.dumps(r['before_types'],ensure_ascii=False)+'；最终：'+json.dumps(r['after_types'],ensure_ascii=False),
      '', '| reviewed状态 | 首次初稿/72 | 最终稿/72 |','|---|---:|---:|',
      *[f"| {s} | {r['initial_states'][s]} ({r['initial_states'][s]/72:.2%}) | {a['agent']['reviewed']['states'][s]} ({a['agent']['reviewed']['state_fractions'][s]:.2%}) |" for s in STATES],
      '', '## 解码：全部35组结果','',
      'ROUGE仅为无词干化ROUGE-L/LCS F1；重复率为每条输出的重复三元组比例再取均值，不是“多少样本出错”。技术文本无摘要参考。显存为CUDA allocated：先对每次运行取峰值，再求平均；另列观察到的最大值，非reserved／整卡占用。',
      '', '| 模型 | 数据/划分 | 方法 | n | ROUGE-L | FactKB | 重复率 | 延迟s | 输出token | 平均峰值MiB | 最大峰值MiB |',
      '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in a['decoding']:
        rouge='—' if r['rouge_l'] is None else f"{r['rouge_l']:.4f}"
        lines.append(f"| {r['model'].split('/')[-1]} | {r['dataset']}/{r['split']} | {r['method']} | {r['count']} | {rouge} | {r['factkb_mean']:.4f} | {r['repeated_trigram_fraction']:.2%} | {r['seconds']:.3f} | {r['output_tokens']:.2f} | {r['peak_mib']:.3f} | {r['max_observed_peak_mib']:.3f} |")
    lines+=['','0.5B只覆盖固定50条XSum测试样本，不能与1.5B的200条简单排名。1MiB=2^20 bytes，1GiB=1024MiB。',
             '', '## 同输入／预算的公式版与源码对齐版开发对照','',
             '| 子集/方法 | n | 原版重复率 | 对齐版重复率 | 原版EOS率 | 对齐版EOS率 |','|---|---:|---:|---:|---:|---:|']
    for key,r in a['comparison'].items():lines.append(f"| {key} | {r['n']} | {r['before_rep']:.2%} | {r['after_rep']:.2%} | {r['before_eos']:.1%} | {r['after_eos']:.1%} |")
    t=a['training'];lines+=['','## 已有后训练证据','',f"模型{t['model']}，LoRA SFT／DPO各{t['sft_steps']}／{t['dpo_steps']}步，{t['trainable_parameters']:,}可训练参数；{t['preference_pairs']}对偏好数据含{t['same_answer_invalid_evidence_pairs']}对同答案证据无效负例。",
      '固定工具观测、同来源4题字段留出：base 0/4→SFT 4/4→DPO 0/4。SimPO仅损失与梯度验证。此证据来自历史工程，不表示本轮CFB生成数据已做后训练。','']
    return '\n'.join(lines)


if __name__=='__main__':
    audit=build_audit();write_json(ROOT/'artifacts/paper/resume_metric_audit.json',audit)
    text=render(audit);(ROOT/'docs/14_简历重构指标核对.md').write_text(text,encoding='utf-8')
    print(text)
