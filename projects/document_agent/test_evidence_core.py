from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from evidence_core import (Claim, Corpus, SearchArgs, Task, TraceStore, audit_claim,
                           contains_value, filter_trajectories, supervision_roles, call_tool)
from run_project import ROOT, gold_cases, messages_for


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = Corpus.load(ROOT / 'artifacts' / 'corpus.json')
        cls.gold = {r['id']: r for r in gold_cases(cls.corpus)}

    def record(self, identifier='outer_diameter_subject'):
        return copy.deepcopy(self.gold[identifier])

    def test_real_merged_header_uses_geometry(self):
        # The header is c2 but its merged data cell is c1 in this actual PDF.
        element = self.corpus.by_id['K231105:p7:t1:r7:c1']
        self.assertEqual(element.scope, 'subject')
        self.assertIn('Subject Device', element.context)

    def test_scanned_pages_are_not_silent_empty_evidence(self):
        pages = [p for p in self.corpus.pages if p['document_id'] == 'K243708' and p['pdf_page'] == 6]
        self.assertTrue(pages[0]['needs_ocr_review'])

    def test_true_source_record(self):
        r = self.record()
        self.assertTrue(audit_claim(self.corpus, Task(**r['task']), r['claim'])['accepted'])

    def test_same_numeric_value_does_not_hide_wrong_column(self):
        r = self.record('inner_diameter_subject')
        r['claim'] = self.record('inner_diameter_predicate')['claim']
        verdict = audit_claim(self.corpus, Task(**r['task']), r['claim'])
        self.assertFalse(verdict['accepted'])
        self.assertIn('wrong_or_unknown_scope', verdict['errors'])

    def test_numeric_prefix_is_not_support(self):
        self.assertFalse(contains_value('5', 'Max. 5.85 mm'))
        self.assertFalse(contains_value('5.8', 'Max. 5.85 mm'))
        self.assertTrue(contains_value('5.0', 'Max. 5.0 mm'))

    def test_number_adjacent_to_unit(self):
        self.assertTrue(contains_value('380', '380mm'))
        self.assertFalse(contains_value('38', '380mm'))

    def test_missing_maximum_qualifier(self):
        for quote in ['Max. 5.0 mm', '5.0 mm']:
            r = self.record()
            r['claim']['qualifier'] = None
            r['claim']['evidence'][0]['quote'] = quote
            self.assertIn('missing_or_wrong_bound_qualifier', audit_claim(self.corpus, Task(**r['task']), r['claim'])['errors'])

    def test_wrong_page_rejected(self):
        r = self.record()
        r['claim']['evidence'][0]['pdf_page'] = 3
        self.assertIn('wrong_page', audit_claim(self.corpus, Task(**r['task']), r['claim'])['errors'])

    def test_invented_source_quote_rejected(self):
        r = self.record()
        r['claim']['evidence'][0]['quote'] = '5.0 mm according to an invented note'
        self.assertIn('quote_not_in_source', audit_claim(self.corpus, Task(**r['task']), r['claim'])['errors'])

    def test_cross_document_reference_rejected(self):
        r = self.record()
        e = next(e for e in self.corpus.elements if e.document_id == 'K252598')
        r['claim']['evidence'] = [{'element_id':e.element_id, 'pdf_page':e.pdf_page, 'quote':e.text}]
        self.assertIn('wrong_document', audit_claim(self.corpus, Task(**r['task']), r['claim'])['errors'])

    def test_extra_schema_field_and_invalid_supported_state(self):
        r = self.record()
        r['claim']['regulatory_approval'] = True
        self.assertIn('schema_error', audit_claim(self.corpus, Task(**r['task']), r['claim'])['errors'])
        with self.assertRaises(ValidationError):
            Claim(field='outer_diameter',status='supported',value='5.0',evidence=[])

    def test_cannot_invent_value_for_missing_status(self):
        with self.assertRaises(ValidationError):
            Claim(field='outer_diameter',status='missing',value='5.0')

    def test_top_k_budget_is_bounded(self):
        with self.assertRaises(ValidationError):
            SearchArgs(query='diameter',top_k=100000)

    def test_dispatch_rejects_cross_document_search(self):
        task = Task(**self.record()['task'])
        result = call_tool(self.corpus, task, 'search_evidence', {'query':'diameter','document_id':'K252598'})
        self.assertEqual(result['error'], 'document_scope_violation')

    def test_dispatch_rejects_arbitrary_tool_and_extra_parameters(self):
        task = Task(**self.record()['task'])
        self.assertEqual(call_tool(self.corpus, task, 'shell', {'command':'anything'})['error'], 'unknown_tool')
        self.assertEqual(call_tool(self.corpus, task, 'read_element', {'element_id':'x','extra':'y'})['error'], 'invalid_arguments')

    def test_dispatch_exposes_ocr_review_flags(self):
        task = Task(task_id='scan',document_id='K243708',field='device_name',query='device name',scope='any')
        result = call_tool(self.corpus, task, 'list_pages', {})
        self.assertTrue(result['ok'])
        self.assertTrue(any(p['needs_ocr_review'] for p in result['data']))

    def test_trace_can_reopen_persisted_events(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'traces.sqlite'
            first = TraceStore(path)
            run = first.start(Task(**self.record()['task']))
            first.append(run, {'tool':'read_element','result':{'page':7}})
            first.db.close()
            second = TraceStore(path)
            self.assertEqual(second.read(run), [{'tool':'read_element','result':{'page':7}}])
            second.db.close()

    def test_tool_observation_mismatch_not_kept_for_sft(self):
        r = self.record()
        r['answer_matches_expected'] = True
        r['messages'] = messages_for(r)
        r['messages'][3]['content'] = 'Tool actually returned something else'
        kept, rejected = filter_trajectories(self.corpus, [r])
        self.assertFalse(kept)
        self.assertIn('observation_mismatch', rejected[0]['errors'])

    def test_unobserved_but_correct_evidence_not_kept(self):
        r = self.record()
        r['answer_matches_expected'] = True
        r['messages'] = messages_for(r)
        r['messages'] = [m for m in r['messages'] if m['role'] != 'tool']
        kept, rejected = filter_trajectories(self.corpus, [r])
        self.assertFalse(kept)
        self.assertIn('unobserved_evidence', rejected[0]['errors'])

    def test_role_boundaries_exclude_tool_and_user(self):
        rows = supervision_roles(messages_for(self.record()))
        self.assertEqual([r['supervise'] for r in rows], [False,False,True,False,True])


if __name__ == '__main__':
    unittest.main(verbosity=2)
