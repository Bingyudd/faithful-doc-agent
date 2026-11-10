import copy
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from evaluate_live import SYSTEM_PROMPT, score_answer, summarize
from live_provider import ChatCompletionsProvider, ProviderError


class LiveProviderTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.status = 200
        self.response = {'choices': [{'message': {'role': 'assistant', 'content': '{}'}}]}
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                owner.requests.append((self.headers.get('Authorization'), body))
                self.send_response(owner.status)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(owner.response).encode())

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.provider = ChatCompletionsProvider(
            f'http://127.0.0.1:{self.server.server_port}/v1', 'test-model', 'unit-test-secret')

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_wire_payload_and_credentials_not_logged(self):
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': '{}'}]
        original = copy.deepcopy(messages)
        self.provider(messages, [])
        self.assertEqual(messages, original)
        header, payload = self.requests[0]
        self.assertEqual(header, 'Bearer unit-test-secret')
        self.assertEqual(payload['model'], 'test-model')
        self.assertEqual(payload['tool_choice'], 'auto')
        self.assertFalse(payload['stream'])
        self.assertNotIn('unit-test-secret', json.dumps(self.provider.exchanges))

    def test_auth_failure_is_not_retried_or_body_exposed(self):
        self.status, self.response = 401, {'error': 'unit-test-secret'}
        with self.assertRaisesRegex(ProviderError, 'HTTP 401'):
            self.provider([], [])
        self.assertEqual(len(self.requests), 1)
        self.assertNotIn('unit-test-secret', json.dumps(self.provider.exchanges))

    def test_malformed_response_is_recorded_as_failure(self):
        self.response = {'choices': []}
        with self.assertRaises(ProviderError):
            self.provider([], [])
        self.assertIn('error', self.provider.exchanges[0])

    def test_reasoning_and_tool_arguments_preserved(self):
        message = {'role': 'assistant', 'content': None, 'reasoning_content': 'inspect source',
                   'tool_calls': [{'id': 'c1', 'type': 'function', 'function': {
                       'name': 'search_evidence', 'arguments': '{"query":"diameter"}'}}]}
        self.response = {'choices': [{'message': message}]}
        self.assertEqual(self.provider([], []), message)

    def test_gateway_echo_of_key_redacted(self):
        self.response['choices'][0]['message']['content'] = 'unit-test-secret'
        self.assertEqual(self.provider([], [])['content'], '[REDACTED]')
        self.assertNotIn('unit-test-secret', json.dumps(self.provider.exchanges))


class LiveScoringTests(unittest.TestCase):
    def setUp(self):
        self.expected = {'field': 'outer_diameter', 'status': 'supported', 'value': '5.0',
                         'unit': 'mm', 'qualifier': 'max',
                         'evidence': [{'element_id': 'e1', 'pdf_page': 1, 'quote': 'Max 5.0 mm'}]}

    def score(self, claim, status='validated'):
        return score_answer({'status': status, 'messages': [
            {'role': 'assistant', 'content': json.dumps(claim)}]}, self.expected)

    def test_provenance_acceptance_alone_does_not_pass(self):
        claim = dict(self.expected, value='5.85')
        self.assertFalse(self.score(claim)['e2e_pass'])

    def test_missing_bound_or_unit_does_not_pass(self):
        for key in ('unit', 'qualifier'):
            self.assertFalse(self.score(dict(self.expected, **{key: None}))['e2e_pass'])

    def test_correct_answer_needs_observed_valid_evidence(self):
        self.assertTrue(self.score(self.expected)['e2e_pass'])
        self.assertFalse(self.score(self.expected, 'needs_review')['e2e_pass'])

    def test_malformed_claim_is_failure_not_crash(self):
        for payload in ([], None, {'evidence': [None]}, {'status': 'supported'}):
            self.assertFalse(self.score(payload)['valid_final_schema'])

    def test_failures_remain_in_denominator(self):
        rows = [{'status': status, 'e2e_pass': passed, 'tool_calls': 2, 'model_requests': 3,
                 'tool_errors': 0, 'audit_rejections': 0, 'final_errors': []}
                for status, passed in [('validated', True), ('provider_error', False)]]
        summary = summarize(rows)
        self.assertEqual(summary['completed_tasks'], 2)
        self.assertEqual(summary['e2e_pass_rate'], 0.5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
