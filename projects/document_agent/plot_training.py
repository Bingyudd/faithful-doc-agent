"""Plot measured optimizer-step losses without smoothing or invented error bars."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / 'docs/experiments/2026-11-06'


def main():
    rows = list(csv.DictReader((FOLDER / 'training_steps.csv').open(encoding='utf-8')))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), layout='constrained')
    for ax, phase, title, color in zip(axes, ('sft', 'dpo'),
                                      ('SFT: assistant-token CE', 'DPO: reference-corrected loss'),
                                      ('#2563A6', '#B15B36')):
        subset = [row for row in rows if row['phase'] == phase]
        ax.plot([int(r['step']) for r in subset], [float(r['loss']) for r in subset], color=color, linewidth=1.6)
        ax.set(title=title, xlabel='Optimizer step', ylabel='Training loss', xlim=(1, 40))
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=0.2)
    fig.suptitle('Qwen2.5-0.5B LoRA | 16 training tasks | one seeded run', fontsize=11)
    fig.savefig(FOLDER / 'training_loss.png', dpi=180)
    fig.savefig(FOLDER / 'training_loss.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
