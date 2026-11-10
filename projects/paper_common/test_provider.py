import unittest
from unittest.mock import patch
from .snapshot_provider import SnapshotProvider
from .provider import ProviderError


class ProviderTests(unittest.TestCase):
    def test_message_and_tool_log_are_snapshots(self):
        p=SnapshotProvider(api_key='unit-test-secret')
        p._request=lambda path,payload:{'choices':[{'message':{'role':'assistant','content':'ok'},'finish_reason':'stop'}]}
        messages=[{'role':'user','content':'original'}]
        tools=[{'type':'function','function':{'name':'read','parameters':{'type':'object'}}}]
        p.chat(messages,tools)
        messages[0]['content']='changed';messages.append({'role':'assistant','content':'future'})
        tools[0]['function']['name']='changed'
        self.assertEqual(p.exchanges[0]['request']['messages'],[{'role':'user','content':'original'}])
        self.assertEqual(p.exchanges[0]['request']['tools'][0]['function']['name'],'read')

    def test_response_secret_redaction(self):
        p=SnapshotProvider(api_key='unit-test-secret')
        p._request=lambda path,payload:{'choices':[{'message':{'role':'assistant','content':'unit-test-secret'},'finish_reason':'stop'}]}
        result=p.chat([{'role':'user','content':'hello'}])
        self.assertEqual(result['content'],'[REDACTED]')
        self.assertNotIn('unit-test-secret',str(p.exchanges))

    def test_authentication_is_not_retried(self):
        p=SnapshotProvider(api_key='unit-test-secret')
        def fail(path,payload):raise ProviderError('HTTP 401 from configured endpoint')
        p._request=fail
        with self.assertRaises(ProviderError):p.chat([])
        self.assertEqual(len(p.exchanges),1)

    def test_transient_retries_are_bounded(self):
        p=SnapshotProvider(api_key='unit-test-secret',retries=2)
        def fail(path,payload):raise ProviderError('HTTP 503 from configured endpoint')
        p._request=fail
        with patch('projects.paper_common.provider.time.sleep'):
            with self.assertRaises(ProviderError):p.chat([])
        self.assertEqual(len(p.exchanges),3)


if __name__=='__main__':unittest.main()
