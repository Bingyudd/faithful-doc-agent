"""Local UI smoke tests exercise saved reports without invoking the model API."""
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest
from projects.paper_common.io import ROOT


class UITests(unittest.TestCase):
    def test_start_and_upload_switch(self):
        app=AppTest.from_file(str(Path(__file__).with_name('app.py'))).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value,'技术文档审校台')
        next(x for x in app.radio if x.label=='资料来源').set_value('上传资料').run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(next(x for x in app.button if x.label=='生成初稿').disabled)

    def test_open_saved_report(self):
        if not list((ROOT/'artifacts/paper/raw/reports_v2').glob('*/*/result.json')):
            self.skipTest('No live report yet')
        app=AppTest.from_file(str(Path(__file__).with_name('app.py'))).run(timeout=30)
        button=next((x for x in app.button if x.label=='打开报告'),None)
        self.assertIsNotNone(button)
        button.click().run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(app.text_area)

    def test_manual_review_navigation_is_read_only(self):
        app=AppTest.from_file(str(Path(__file__).with_name('app.py'))).run(timeout=30)
        next(x for x in app.radio if x.label=='工作区').set_value('抽样复核').run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(app.header[0].value,'固定抽样复核')


if __name__=='__main__':unittest.main()
