"""Snapshot caller-owned messages before the legacy transport records them.

Added after the frozen campaign. It changes log ownership, not the serialized
request at send time. Archived responses and business events remain authoritative;
archived mutable request histories must not be described as exact wire captures.
"""
import copy
from .provider import Provider


class SnapshotProvider(Provider):
    def chat(self,messages,tools=None):
        return super().chat(copy.deepcopy(messages),copy.deepcopy(tools))
