"""Reuse the existing loopback-only transport, adding bounded retry and usage."""
from __future__ import annotations

import json
import os
import time

from projects.document_agent.live_provider import ChatCompletionsProvider, ProviderError


class Provider(ChatCompletionsProvider):
    def __init__(self, model='deepseek-v4.1-flash', base_url='http://127.0.0.1:8788/v1',
                 api_key=None, max_tokens=16384, timeout=180, retries=2):
        super().__init__(base_url, model, api_key or os.environ.get('LOCAL_LLM_API_KEY', ''),
                         timeout=timeout, max_tokens=max_tokens)
        self.retries = retries

    def chat(self, messages, tools=None):
        payload = dict(model=self.model, messages=messages, temperature=0,
                       max_tokens=self.max_tokens, stream=False)
        if tools:
            payload.update(tools=tools, tool_choice='auto')
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            event = {'request': payload, 'attempt': attempt}
            self.exchanges.append(event)
            try:
                response = self._request('/chat/completions', payload)
                if self._key:
                    response = json.loads(json.dumps(response).replace(
                        json.dumps(self._key)[1:-1], '[REDACTED]'))
                event['response'] = response
                message = response['choices'][0]['message']
                if response['choices'][0].get('finish_reason') == 'length':
                    raise ValueError('Model output exceeded token budget')
                return {k: v for k, v in message.items()
                        if k in ('role', 'content', 'tool_calls', 'reasoning_content')}
            except (ProviderError, KeyError, IndexError) as exc:
                event['error'] = type(exc).__name__ + ': ' + str(exc)
                if attempt == self.retries or any(code in str(exc) for code in ('HTTP 401','HTTP 403','HTTP 404')):
                    raise
                time.sleep(min(2 ** attempt, 4))
            finally:
                event['elapsed_seconds'] = time.perf_counter() - started

    def stats(self):
        totals = {'requests': len(self.exchanges), 'prompt_tokens': 0,
                  'completion_tokens': 0, 'seconds': 0.0, 'reported_models': []}
        for event in self.exchanges:
            totals['seconds'] += event['elapsed_seconds']
            response = event.get('response', {})
            for name in ('prompt_tokens', 'completion_tokens'):
                totals[name] += response.get('usage', {}).get(name, 0) or 0
            model = response.get('model')
            if model and model not in totals['reported_models']:
                totals['reported_models'].append(model)
        return totals
