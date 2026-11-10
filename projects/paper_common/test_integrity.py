import unittest
from projects.paper_common.io import ROOT,read_json,read_jsonl,digest
from projects.paper_common.evaluate import rouge_l,text_metrics
from tools.paper_skills import compare_claim


class IntegrityTests(unittest.TestCase):
    def test_source_families_and_manifest(self):
        manifest=read_json(ROOT/'data/paper/sources.json');sources=manifest['sources']
        self.assertEqual(len(sources),20)
        self.assertEqual(len({d['family'] for d in sources}),20)
        self.assertEqual(sum(d['split']=='dev' for d in sources),8)
        self.assertEqual(sum(d['split']=='test' for d in sources),12)
        self.assertEqual(digest([(d['id'],d['split']) for d in sources]),manifest['split_hash'])
        self.assertFalse({'K231105','K243708','K252598'}&{d['id'] for d in sources})

    def test_xsum_content_ids_and_hashes(self):
        rows=read_jsonl(ROOT/'data/paper/xsum_dev.jsonl')+read_jsonl(ROOT/'data/paper/xsum_test.jsonl')
        manifest=read_json(ROOT/'data/paper/xsum_manifest.json')
        self.assertEqual(len(rows),250);self.assertEqual(len({r['id'] for r in rows}),250)
        self.assertEqual(digest(rows),manifest['sha256'])
        self.assertEqual([(r['id'],digest(r)) for r in rows],[(r['id'],r['sha256']) for r in manifest['samples']])

    def test_transfer_matches_sources(self):
        sources={d['id']:d for d in read_json(ROOT/'data/paper/sources.json')['sources']}
        corpus={e['id']:e for e in read_json(ROOT/'data/paper/corpus.json')['elements']}
        for split,count in [('dev',40),('test',60)]:
            rows=read_jsonl(ROOT/f'data/paper/technical_{split}.jsonl');self.assertEqual(len(rows),count)
            for row in rows:
                self.assertEqual(row['source'],corpus[row['source_id']]['text'])
                self.assertEqual(row['split'],sources[row['document_id']]['split'])

    def test_metrics_have_known_values(self):
        self.assertEqual(rouge_l('a b c','a b c'),1)
        self.assertAlmostEqual(rouge_l('a b','a c'),.5)
        self.assertEqual(text_metrics('Length is 10 mm.','Length is 10 cm.')['numeric_precision'],0)

    def test_metric_conditions_and_finite_gate(self):
        c=dict(confirmed=True,value=1,metric='accuracy',model='m',revision='r',split_hash='s',config_hash='c',unit='fraction',run_id='id')
        self.assertEqual(compare_claim(c,c)['status'],'match')
        self.assertEqual(compare_claim(c,{**c,'unit':'percent'})['status'],'incomparable')
        self.assertEqual(compare_claim(c,{**c,'value':float('nan')})['status'],'unverified')
        self.assertEqual(compare_claim(c,{**c,'value':None})['status'],'unverified')
        self.assertEqual(compare_claim(c,{**c,'metric':'recall'})['status'],'incomparable')


if __name__=='__main__':unittest.main()
