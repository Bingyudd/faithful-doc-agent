import copy
import json
import tempfile
import unittest
from pathlib import Path

from evidence_core import Corpus, Task, TraceStore
from run_project import ROOT, gold_cases
from tool_protocol import ProtocolError, check_messages, function_tools, reexecute_trace, run_tool_loop
from verify_protocol import scripted_provider


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = Corpus.load(ROOT / 'artifacts/corpus.json')
        cls.record = gold_cases(cls.corpus)[0]

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.store = TraceStore(Path(self.folder.name)/'trace.sqlite')
        self.task = Task(**self.record['task'])

    def tearDown(self):
        self.store.db.close()
        self.folder.cleanup()

    def call(self, identifier='call_one', arguments=None):
        args = arguments if arguments is not None else json.dumps({'element_id':self.record['claim']['evidence'][0]['element_id']})
        return {'id':identifier,'type':'function','function':{'name':'read_element','arguments':args}}

    def final(self):
        return {'role':'assistant','content':json.dumps(self.record['claim'])}

    def test_function_tool_descriptors(self):
        tools = function_tools()
        self.assertEqual(len(tools),4)
        self.assertTrue(all(t['type']=='function' and 'parameters' in t['function'] for t in tools))

    def test_loop_matches_call_and_response_ids(self):
        result = run_tool_loop(self.corpus,self.task,scripted_provider(self.record),self.store)
        self.assertEqual(result['status'],'validated')
        self.assertEqual(result['messages'][2]['tool_calls'][0]['id'],result['messages'][3]['tool_call_id'])
        self.assertTrue(check_messages(result['messages']))

    def test_correct_but_unread_evidence_not_accepted(self):
        result = run_tool_loop(self.corpus,self.task,lambda m,t:self.final(),self.store)
        self.assertEqual(result['status'],'needs_review')
        self.assertIn('unobserved_evidence',result['verdict']['errors'])

    def test_multiple_tool_calls(self):
        def provider(messages, tools):
            return self.final() if messages[-1]['role']=='tool' else {'role':'assistant','content':None,'tool_calls':[self.call('a'),self.call('b')]}
        result = run_tool_loop(self.corpus,self.task,provider,self.store)
        self.assertEqual(result['tool_calls'],2)
        self.assertEqual(result['status'],'validated')

    def test_malformed_arguments_return_tool_error(self):
        def provider(messages, tools):
            return self.final() if messages[-1]['role']=='tool' else {'role':'assistant','content':None,'tool_calls':[self.call(arguments='{broken')]}
        result = run_tool_loop(self.corpus,self.task,provider,self.store)
        self.assertEqual(json.loads(result['messages'][3]['content'])['error'],'invalid_json_arguments')
        self.assertEqual(result['status'],'needs_review')

    def test_tool_budget_prevents_partial_batch(self):
        provider = lambda m,t: {'role':'assistant','content':None,'tool_calls':[self.call('a'),self.call('b')]}
        result = run_tool_loop(self.corpus,self.task,provider,self.store,max_tool_calls=1)
        self.assertEqual(result['status'],'tool_budget')
        self.assertEqual(result['tool_calls'],0)
        self.assertTrue(check_messages(result['messages']))

    def test_step_budget_stops_loop(self):
        provider = lambda m,t: {'role':'assistant','content':None,'tool_calls':[self.call('call_'+str(len(m)))]}
        result = run_tool_loop(self.corpus,self.task,provider,self.store,max_steps=2)
        self.assertEqual(result['status'],'step_budget')
        self.assertEqual(result['tool_calls'],2)

    def test_orphan_and_reused_ids_rejected(self):
        with self.assertRaises(ProtocolError):
            check_messages([{'role':'tool','tool_call_id':'orphan','content':'x'}])
        pair=[{'role':'assistant','content':None,'tool_calls':[self.call()]},
              {'role':'tool','tool_call_id':'call_one','content':'x'}]
        with self.assertRaises(ProtocolError):
            check_messages(pair + pair)

    def test_actual_tool_reexecution_and_fingerprint_guard(self):
        result = run_tool_loop(self.corpus,self.task,scripted_provider(self.record),self.store)
        self.assertTrue(all(row['matches_recorded'] for row in reexecute_trace(self.corpus,self.store,result['run_id'])))
        changed = Corpus(self.corpus.elements,self.corpus.pages,copy.deepcopy(self.corpus.documents))
        next(d for d in changed.documents if d['document_id']==self.task.document_id)['sha256']='changed'
        with self.assertRaises(ValueError):
            reexecute_trace(changed,self.store,result['run_id'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
