import tempfile
import unittest
from pathlib import Path
from .schema import Section,Report,TEMPLATES
from .evidence import audit_section,retrieve
from .engine import ReportEngine,BudgetExceeded
from tools.paper_skills import compare_claim

SOURCE={'id':'D:p1:c0','document_id':'D','page':1,'text':'The subject device has a working length of 10 mm and is single use.'}


def section(**change):
    value={'id':'identity','title':'Identity','status':'supported','statements':[{'text':'The working length is 10 mm.',
           'target_document':'D','citations':[{'source_id':SOURCE['id'],'quote':'working length of 10 mm'}]}],'notes':[]}
    value.update(change)
    return Section.model_validate(value)


class Fake:
    model='fixture';max_tokens=4096
    def __init__(self):self.exchanges=[]
    def stats(self):return {'requests':len(self.exchanges),'prompt_tokens':0,'completion_tokens':0,'seconds':0,'reported_models':[]}
    def chat(self,messages,tools=None):
        import json
        self.exchanges.append({'fixture':True})
        prompt=json.loads(messages[-1]['content'])
        spec=prompt['section']
        return {'role':'assistant','content':section(id=spec['id'],title=spec['title']).model_dump_json()}


class AgentTests(unittest.TestCase):
    def test_correct_quote(self):self.assertEqual(audit_section(section(),{SOURCE['id']:SOURCE},'D'),[])
    def test_forged_quote(self):
        s=section();s.statements[0].citations[0].quote='a working length of 20 mm'
        self.assertIn('quote_not_in_source',{x['code'] for x in audit_section(s,{SOURCE['id']:SOURCE},'D')})
    def test_wrong_product(self):
        s=section();s.statements[0].target_document='X'
        self.assertEqual(audit_section(s,{SOURCE['id']:SOURCE},'D')[0]['code'],'wrong_product_document')
    def test_unobserved_citation(self):
        self.assertIn('unobserved_citation',{x['code'] for x in audit_section(section(),{},'D')})
    def test_wrong_numeric_unit(self):
        s=section();s.statements[0].text='The working length is 10 cm.'
        self.assertIn('unsupported_number_or_unit',{x['code'] for x in audit_section(s,{SOURCE['id']:SOURCE},'D')})
    def test_missing_and_conflict_contract(self):
        with self.assertRaises(ValueError):section(status='missing')
        with self.assertRaises(ValueError):section(status='conflict')
    def test_search_document_isolation(self):
        self.assertEqual(retrieve([SOURCE],'working length','X'),[])
        self.assertEqual(retrieve([SOURCE],'working length','D')[0]['id'],SOURCE['id'])
    def test_budget(self):
        with self.assertRaises(BudgetExceeded):ReportEngine(Fake(),[SOURCE],max_calls=0).ask('test')
    def test_completed_run_cached(self):
        with tempfile.TemporaryDirectory() as d:
            fake=Fake();doc={'id':'D','device_name':'demo'}
            first=ReportEngine(fake,[SOURCE]).run(doc,'overview','decomposed',d)
            self.assertEqual(first['status'],'complete')
            self.assertEqual(len(fake.exchanges),3)
            again=ReportEngine(fake,[SOURCE]).run(doc,'overview','decomposed',d)
            self.assertEqual(first,again);self.assertEqual(len(fake.exchanges),3)
    def test_resume_partial_sections(self):
        with tempfile.TemporaryDirectory() as d:
            fake=Fake();doc={'id':'D','device_name':'demo'}
            first=ReportEngine(fake,[SOURCE]).run(doc,'overview','decomposed',d)
            directory=next(Path(d).iterdir())
            (directory/'result.json').unlink()
            import json
            state=json.loads((directory/'state.json').read_text())
            state['sections']=state['sections'][:1];state['calls']=1
            (directory/'state.json').write_text(json.dumps(state))
            resumed=Fake();result=ReportEngine(resumed,[SOURCE]).run(doc,'overview','decomposed',d)
            self.assertEqual(result['status'],'complete');self.assertEqual(len(resumed.exchanges),2)
    def test_claim_requires_confirmed_and_conditions(self):
        self.assertEqual(compare_claim({'value':1,'confirmed':False},{'value':1})['status'],'unverified')
        self.assertEqual(compare_claim({'value':1,'confirmed':True},{'value':1})['status'],'unverified')

    def test_json_repair_is_bounded(self):
        class Malformed(Fake):
            def chat(self,messages,tools=None):
                self.exchanges.append({'fixture':True})
                return {'role':'assistant','content':'not json' if len(self.exchanges)==1 else section().model_dump_json()}
        fake=Malformed();engine=ReportEngine(fake,[SOURCE])
        self.assertEqual(engine.object('test',Section).id,'identity')
        self.assertEqual(engine.calls,2)

    def test_repeated_json_failure_stops(self):
        class Invalid(Fake):
            def chat(self,messages,tools=None):return {'role':'assistant','content':'not json'}
        engine=ReportEngine(Invalid(),[SOURCE])
        with self.assertRaises(ValueError):engine.object('test',Section)
        self.assertEqual(engine.calls,2)


if __name__=='__main__':unittest.main()
