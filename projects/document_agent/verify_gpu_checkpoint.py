"""Reload both saved adapters and compare held-out response scores to run evidence."""
import argparse
import gc
import hashlib
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_lora import MODEL, REVISION, prepare_data, target_logp


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    report = json.loads((args.run / 'report.json').read_text(encoding='utf-8'))
    assert report['complete'] and report['optimizer_steps'] == 80
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    pairs, _ = prepare_data(tokenizer)
    selected = [next(p for p in pairs if p['split'] == split) for split in ('train', 'holdout')]
    checks = []
    torch.set_num_threads(4)
    for stage in ('sft', 'dpo'):
        base = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
                                                  torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
        folder = args.run / (stage + '_adapter')
        model = PeftModel.from_pretrained(base, folder, is_trainable=False).eval()
        expected = {r['id']: r for r in report[stage]['rows']}
        diffs = []
        with torch.no_grad():
            for pair in selected:
                for side in ('chosen', 'rejected'):
                    actual, _ = target_logp(model, pair[side + '_encoded'])
                    diff = abs(actual.item() - expected[pair['id']][side + '_logp'])
                    diffs.append(diff)
                    if diff > 0.02:
                        raise AssertionError((stage, pair['id'], side, diff))
        checks.append({'stage': stage, 'response_scores_checked': len(diffs),
                       'max_absolute_logp_difference': max(diffs),
                       'adapter_sha256': hashlib.sha256((folder / 'adapter_model.safetensors').read_bytes()).hexdigest()})
        del model, base
        gc.collect()
        torch.cuda.empty_cache()
    result = {'passed': True, 'reload_checks': checks}
    (args.run / 'checkpoint_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
