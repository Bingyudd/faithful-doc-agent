"""Assemble a publishable bundle: allowlisted code/docs plus a generated results table.

The bundle is written to `publish/bundle/` (git-ignored) so the repository history keeps
only the sources. Personal material (resumes, contact details, shared notes, raw runs and
adapters) is never copied.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from projects.paper_common.io import ROOT, read_json

# files or globs copied as-is (relative to the repository root)
ALLOW = [
    'projects/report_agent/*.py',
    'projects/report_agent/README.md',
    'projects/faithful_decoding/*.py',
    'projects/faithful_decoding/README.md',
    'projects/paper_common/*.py',
    'projects/document_agent/*.py',
    'projects/document_agent/README.md',
    'configs/*.json',
    'data/paper/*.json',
    'data/paper/*.jsonl',
    'data/paper/*.md',
    'artifacts/paper/*_report.json',
    'artifacts/paper/examples/reviewed_report.md',
    'artifacts/paper/examples/reviewed_report.json',
    'tools/paper_skills.py',
    'tools/validate_paper_delivery.py',
    'tools/source_quality.py',
    'tools/write_paper_report.py',
    'tools/resume_metric_audit.py',
    'tools/build_interview_factsheet.py',
    'tools/build_publish_bundle.py',
    'environment-paper.yml',
    'environment-paper.lock.txt',
    '.gitattributes',
]

# never copied, even if a pattern above would match
DENY = [
    'resumes/', 'memory/', 'output/', 'tmp/', 'interview_prep/', 'docs/experiments/',
    'artifacts/paper/raw/', 'artifacts/paper/ui/', 'projects/document_agent/artifacts/',
    'projects/document_agent/data/', 'provenance.json',
]

PHONE = re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)')
EMAIL = re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+')


def collect():
    picked = []
    for pattern in ALLOW:
        for path in sorted(ROOT.glob(pattern)):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT).as_posix()
            if any(relative.startswith(prefix) for prefix in DENY):
                continue
            picked.append(path)
    return picked


def results_markdown(audit):
    agent = audit['agent']
    revision, cross = audit['revision'], audit['cross_strategy']
    decode = audit['decoding_selected']
    training = audit['training']
    main_xsum = decode['main_xsum_context']
    rows = [
        ('直接生成（baseline）', f"{agent['direct']['factkb']:.3f}",
         f"{agent['direct']['citation_pooled']['passed']}/{agent['direct']['citation_pooled']['total']}",
         f"{agent['direct']['no_error_reports']}/24", f"{agent['direct']['requests_mean']:.2f}"),
        ('受控审校管线', f"{agent['reviewed']['factkb']:.3f}",
         f"{agent['reviewed']['citation_pooled']['passed']}/{agent['reviewed']['citation_pooled']['total']}",
         f"{agent['reviewed']['no_error_reports']}/24", f"{agent['reviewed']['requests_mean']:.2f}"),
    ]
    lines = [
        '# 结果总表',
        '',
        '本表由 `tools/build_publish_bundle.py` 从 `artifacts/paper/resume_metric_audit.json` 生成，'
        '每个数字都能用 `python -m tools.resume_metric_audit` 重算。',
        '',
        '## 一、文档审校（DeepSeek，24 份留出报告，12 个来源组）',
        '',
        '| 策略 | FactKB（自动事实评分） | 逐引文通过 | 零告警报告 | 平均请求/报告 |',
        '|---|---|---|---|---|',
    ]
    lines += [f'| {a} | {b} | {c} | {d} | {e} |' for a, b, c, d, e in rows]
    lines += [
        '',
        f"- 总机械告警：{cross['direct_errors']} → {cross['reviewed_errors']}（净降 {cross['error_reduction_fraction']:.1%}，相对直接生成）",
        f"- 审校内部初稿→终稿：{revision['initial_errors']} → {revision['final_errors']}（净降 {revision['net_reduction_fraction']:.0%}，"
        f"修订章节 {revision['revised_sections']}/{revision['sections']}，上限 2 轮）",
        f"- FactKB 只对含事实陈述的报告计分：直接生成 {agent['direct']['factkb_n']} 份、审校 {agent['reviewed']['factkb_n']} 份",
        '',
        '## 二、上下文忠实解码（Qwen2.5-1.5B，XSum 200 条测试）',
        '',
        '| 策略 | Context FactKB | ROUGE-L | 单条耗时 (s) |',
        '|---|---|---|---|',
        f"| Greedy | {decode['main_xsum_greedy']['factkb_mean']:.3f} | {decode['main_xsum_greedy']['rouge_l']:.3f} | {decode['main_xsum_greedy']['seconds']:.2f} |",
        f"| CAD（本方法） | {main_xsum['factkb_mean']:.3f} | {main_xsum['rouge_l']:.3f} | {main_xsum['seconds']:.2f} |",
        '',
        f"- 开发集 50 条：重复 3-gram 比例 {audit['comparison']['xsum_dev_token']['before_rep']:.2%} → "
        f"{audit['comparison']['xsum_dev_token']['after_rep']:.2%}",
        f"- 去语义消融 200 条：重复比例 {decode['main_xsum_no_semantic']['repeated_trigram_fraction']:.2%}",
        f"- FDA 技术文本 60 条：Static 相对 Greedy 的 FactKB {decode['main_technical_greedy']['factkb_mean']:.3f} → "
        f"{decode['main_technical_static']['factkb_mean']:.3f}",
        f"- Token 方案平均峰值 allocated：{decode['main_xsum_token']['mean_peak_gib']:.2f} GiB",
        '',
        '## 三、后训练（Qwen2.5-0.5B LoRA，单卡 8GB）',
        '',
        f"- 数据：{training['preference_pairs']} 对偏好数据，其中 {training['same_answer_invalid_evidence_pairs']} 对为同答案、证据无效的强负例",
        f"- 切分：按字段家族 {training['train_pairs']} 对训练 / {training['holdout_pairs']} 对留出；可训练参数 {training['trainable_parameters']:,}",
        f"- 固定黄金工具观测的留出生成：base {training['generation_passes']['base']['passed']}/4 → "
        f"SFT {training['generation_passes']['sft']['passed']}/4 → DPO {training['generation_passes']['dpo']['passed']}/4",
        f"- SimPO：{training['simpo_scope']}",
        '',
        '## 四、这些数字不能说明什么',
        '',
        '- 全部是**同源小样本**：24 份报告、50／200／60 条解码样本；每档单次运行，没有多种子与显著性检验。',
        '- FactKB 是外部自动评分，不是人工判定；机械告警是规则命中，不等于真实错误率（误报/漏报未标注）。',
        '- 生成指标用的是固定黄金工具观测，不代表自主工具调用的端到端成功率。',
        '- DPO 的负结果原样保留，详见 `publish/FAILURES.md`。',
    ]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'publish/bundle')
    parser.add_argument('--check-personal', action='store_true', default=True)
    args = parser.parse_args()
    audit = read_json(ROOT / 'artifacts/paper/resume_metric_audit.json')
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    files = collect()
    copied = []
    for path in files:
        relative = path.relative_to(ROOT)
        target = args.out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(relative.as_posix())
    for extra in ('README.md', 'ARCHITECTURE.md', 'FAILURES.md', 'REPRODUCE.md'):
        source = ROOT / 'publish' / extra
        if source.is_file():
            shutil.copy2(source, args.out / extra)
            copied.append(f'publish/{extra} → {extra}')
    (args.out / 'RESULTS.md').write_text(results_markdown(audit), encoding='utf-8')
    copied.append('RESULTS.md (generated)')
    deck = ROOT / 'output/slides/邓任_项目讲解.pptx'
    if deck.is_file():
        shutil.copy2(deck, args.out / 'SLIDES.pptx')
        copied.append('SLIDES.pptx (generated deck)')
    leaked = []
    for path in args.out.rglob('*'):
        # 只扫我们自己写的文本；data/ 下是公开监管文档原文，里面的机构邮箱不属于个人信息
        if not path.is_file() or path.suffix not in {'.py', '.md'}:
            continue
        text = path.read_text(encoding='utf-8', errors='ignore')
        leaked += [f'{path.relative_to(args.out)}: {m.group(0)}' for m in PHONE.finditer(text)]
        leaked += [f'{path.relative_to(args.out)}: {m.group(0)}' for m in EMAIL.finditer(text)]
    manifest = {'files': copied, 'personal_matches': leaked, 'results_generated_from': 'artifacts/paper/resume_metric_audit.json'}
    (args.out / 'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'copied': len(copied), 'personal_matches': leaked[:10]}, ensure_ascii=False, indent=2))
    if leaked:
        raise SystemExit('personal data found in bundle')


if __name__ == '__main__':
    main()
