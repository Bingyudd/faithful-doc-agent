import copy
import json
import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from post_training import sequence_log_probs
from train_lora import HOLDOUT_FIELDS, ROOT, field_for, target_logp


class GPUTrainingPreparationTests(unittest.TestCase):
    def test_selected_projection_matches_dense_shift_loss_and_gradients(self):
        torch.manual_seed(7)
        config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=24,
                             num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=1, attention_dropout=0.0)
        sparse = Qwen2ForCausalLM(config).eval()
        dense = copy.deepcopy(sparse)
        row = {'input_ids': [1, 2, 3, 4, 5, 6], 'labels': [-100, -100, 3, -100, 5, 6]}
        actual, count = target_logp(sparse, row)
        logits = dense(torch.tensor([row['input_ids']]), use_cache=False).logits
        expected, lengths = sequence_log_probs(logits, torch.tensor([row['labels']]))
        self.assertEqual(count, lengths.item())
        torch.testing.assert_close(actual, expected[0])
        actual.backward()
        expected.sum().backward()
        for a, b in zip(sparse.parameters(), dense.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)

    def test_field_split_keeps_all_variants_and_product_scopes_together(self):
        pairs = [json.loads(line) for line in (ROOT / 'artifacts/preference_pairs.jsonl')
                 .read_text(encoding='utf-8').splitlines() if line.strip()]
        train = [p for p in pairs if field_for(p) not in HOLDOUT_FIELDS]
        held = [p for p in pairs if field_for(p) in HOLDOUT_FIELDS]
        self.assertEqual((len(train), len(held)), (96, 24))
        self.assertFalse({p['task_id'] for p in train} & {p['task_id'] for p in held})
        for field in HOLDOUT_FIELDS:
            self.assertEqual(sum(field_for(p) == field for p in held), 12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
