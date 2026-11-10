"""Small Chat Completions HTTP provider; secrets never enter experiment logs."""
from __future__ import annotations

import copy
import json
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler


class ProviderError(RuntimeError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ChatCompletionsProvider:
    def __init__(self, base_url, model, api_key, *, timeout=90, max_tokens=2048):
        parsed = urlsplit(base_url)
        if (parsed.scheme not in ('http', 'https') or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError('Expected a credential-free HTTP(S) base URL')
        if parsed.scheme == 'http' and parsed.hostname not in ('localhost', '127.0.0.1', '::1'):
            raise ValueError('Non-loopback endpoints require HTTPS')
        self.base_url = base_url.rstrip('/')
        self.model, self._key = model, api_key
        self.timeout, self.max_tokens = timeout, max_tokens
        self.exchanges = []
        # Local traffic must not be forwarded through an environment proxy.
        self._opener = build_opener(ProxyHandler({}), NoRedirect())

    def _request(self, path, payload=None):
        headers = {'Content-Type': 'application/json'}
        if self._key:
            headers['Authorization'] = 'Bearer ' + self._key
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
        request = Request(self.base_url + path, data=data, headers=headers)
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            # Do not copy server bodies: gateways can echo request credentials.
            raise ProviderError(f'HTTP {exc.code} from configured endpoint') from None
        except (URLError, TimeoutError, OSError):
            raise ProviderError('Endpoint connection failed or timed out') from None
        except (ValueError, UnicodeError):
            raise ProviderError('Endpoint returned invalid JSON') from None
        if not isinstance(result, dict):
            raise ProviderError('Endpoint returned a non-object JSON response')
        return result

    def models(self):
        result = self._request('/models')
        return [row['id'] for row in result.get('data', [])
                if isinstance(row, dict) and isinstance(row.get('id'), str)]

    def __call__(self, messages, tools):
        payload = {'model': self.model, 'messages': copy.deepcopy(messages),
                   'tools': tools, 'tool_choice': 'auto', 'temperature': 0,
                   'max_tokens': self.max_tokens, 'stream': False}
        exchange = {'request': payload}
        self.exchanges.append(exchange)
        start = time.perf_counter()
        try:
            result = self._request('/chat/completions', payload)
            # Redact even a malicious server echo of our credential.
            if self._key:
                result = json.loads(json.dumps(result).replace(
                    json.dumps(self._key)[1:-1], '[REDACTED]'))
            exchange['response'] = result
            choices = result.get('choices')
            if (not isinstance(choices, list) or not choices
                    or not isinstance(choices[0], dict)
                    or not isinstance(choices[0].get('message'), dict)):
                raise ProviderError('Endpoint response is missing choices[0].message')
            message = choices[0]['message']
            # Preserve reasoning_content when the gateway requires it on later turns.
            return {key: value for key, value in message.items()
                    if key in ('role', 'content', 'tool_calls', 'reasoning_content')}
        except ProviderError as exc:
            exchange['error'] = str(exc)
            raise
        finally:
            exchange['elapsed_seconds'] = round(time.perf_counter() - start, 3)
