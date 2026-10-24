"""
A minimal stand-in for mp_api.client, so the Materials Project code paths can
be tested offline and without an API key.
"""

import types
from types import SimpleNamespace
from unittest import mock


class FakeMPRestError(Exception):
    pass


class FakeMPRester:
    """
    Fake mp_api.client.MPRester. Subclass it and set ``docs`` to the summary
    documents that ``materials.summary.search`` should return. The keyword
    arguments of the last search are stored in ``search_kwargs``.
    """

    docs = []
    search_kwargs = None

    def __init__(self, api_key=None, **kwargs):
        self.api_key = api_key
        self.materials = SimpleNamespace(summary=SimpleNamespace(search=self._search))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def _search(self, **kwargs):
        type(self).search_kwargs = kwargs
        return list(self.docs)


def fake_mp_api(mprester_cls):
    """
    Patch sys.modules so that ``from mp_api.client import MPRester`` gives
    mprester_cls and ``from mp_api.client.core import MPRestError`` gives
    FakeMPRestError.
    """
    mp_api = types.ModuleType("mp_api")
    client = types.ModuleType("mp_api.client")
    core = types.ModuleType("mp_api.client.core")
    client.MPRester = mprester_cls
    client.core = core
    core.MPRestError = FakeMPRestError
    mp_api.client = client
    return mock.patch.dict(
        "sys.modules",
        {"mp_api": mp_api, "mp_api.client": client, "mp_api.client.core": core},
    )
