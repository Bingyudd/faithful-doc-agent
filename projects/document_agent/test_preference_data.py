import json
import unittest
from pathlib import Path

from evidence_core import Corpus, Task
from preference_data import build_pairs, rule_reward
from run_project import ROOT
from token_masking import encode_supervised, load_tokenizer


class PreferenceDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus=Corpus.load(ROOT/'artifacts/corpus.json')
        cls.cases=json.loads((ROOT/'artifacts/controlled_validation_cases.json').read_text(encoding='utf-8'))
        cls.conversations=[json.loads(line) for line in (ROOT/'artifacts/sft_messages_function_calling.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        cls.pairs=build_pairs(cls.corpus,cls.cases,cls.conversations)

    def test_expected_pair_and_hard_negative_counts(self):
        self.assertEqual(len(self.pairs),120)
        self.assertEqual(sum(p['hard_negative'] for p in self.pairs),86)
        self.assertEqual(len({p['task_id'] for p in self.pairs}),20)

    def test_chosen_and_rejected_share_external_observations(self):
        originals={row['id']:row for row in self.conversations}
        for pair in self.pairs:
            self.assertEqual(pair['prompt'],originals[pair['task_id']]['messages'][:-1])
            self.assertEqual([m['role'] for m in pair['chosen']],['assistant'])
            self.assertEqual([m['role'] for m in pair['rejected']],['assistant'])

    def test_rewards_have_verified_positive_margin(self):
        for pair in self.pairs:
            self.assertEqual(pair['chosen_reward']['total'],1.0)
            self.assertLess(pair['rejected_reward']['total'],pair['chosen_reward']['total'])
            self.assertIn(pair['rejected_reward']['total'],[0.2,0.3,0.5])

    def test_correct_answer_is_not_sufficient_for_full_reward(self):
        pair=next(p for p in self.pairs if p['hard_negative'])
        self.assertEqual(pair['rejected_reward']['components']['answer'],1.0)
        self.assertEqual(pair['rejected_reward']['components']['evidence'],0.0)

    def test_pairs_have_source_split_group(self):
        self.assertEqual({p['split_group'] for p in self.pairs},{'K231105'})
        self.assertEqual({p['negative_category'] for p in self.pairs},
                         {'wrong_value','wrong_page','unknown_anchor','fabricated_quote','wrong_scope','extra_schema_field'})

    def test_preference_labels_do_not_supervise_assistant_actions_in_prompt(self):
        tokenizer=load_tokenizer()
        pair=self.pairs[0]
        row=encode_supervised(tokenizer,pair['prompt']+pair['chosen'],tools=pair['tools'],
                              supervise_from_message=len(pair['prompt']))
        supervised={owner for owner in row['assistant_message_indices'] if owner is not None}
        self.assertEqual(supervised,{len(pair['prompt'])})
        self.assertGreater(sum(label!=-100 for label in row['labels']),0)


if __name__=='__main__':
    unittest.main(verbosity=2)
