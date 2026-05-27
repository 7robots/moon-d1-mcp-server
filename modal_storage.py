"""Modal-native client_storage adapter for FastMCP's OIDCProxy.

Wraps a named modal.Dict as a py-key-value-aio store so it composes with
FernetEncryptionWrapper and PrefixCollectionsWrapper exactly like the
prior Redis-backed implementation.

Modal expires Dict entries after 7 days of inactivity. Active OAuth state
(refresh tokens used at least once a week) is unaffected; users idle past
that window re-authenticate once.
"""

from __future__ import annotations

import modal
from typing_extensions import override

from key_value.aio._utils.compound import compound_key
from key_value.aio._utils.managed_entry import ManagedEntry
from key_value.aio._utils.serialization import (
    BasicSerializationAdapter,
    SerializationAdapter,
)
from key_value.aio.errors import DeserializationError
from key_value.aio.stores.base import BaseStore


class ModalDictStore(BaseStore):
    """py-key-value-aio store backed by a named modal.Dict."""

    _dict: modal.Dict
    _adapter: SerializationAdapter

    def __init__(
        self,
        *,
        dict_name: str,
        default_collection: str | None = None,
    ) -> None:
        # from_name is not a live method — no I/O until first read/write.
        self._dict = modal.Dict.from_name(dict_name, create_if_missing=True)
        self._adapter = BasicSerializationAdapter()
        super().__init__(default_collection=default_collection, stable_api=True)

    @override
    async def _get_managed_entry(
        self, *, collection: str, key: str
    ) -> ManagedEntry | None:
        combo_key = compound_key(collection=collection, key=key)
        raw = await self._dict.get.aio(combo_key, None)
        if not isinstance(raw, str):
            return None
        try:
            return self._adapter.load_json(json_str=raw)
        except DeserializationError:
            return None

    @override
    async def _put_managed_entry(
        self,
        *,
        collection: str,
        key: str,
        managed_entry: ManagedEntry,
    ) -> None:
        combo_key = compound_key(collection=collection, key=key)
        payload = self._adapter.dump_json(
            entry=managed_entry, key=key, collection=collection
        )
        await self._dict.put.aio(combo_key, payload)

    @override
    async def _delete_managed_entry(
        self, *, key: str, collection: str
    ) -> bool:
        combo_key = compound_key(collection=collection, key=key)
        try:
            await self._dict.pop.aio(combo_key)
        except KeyError:
            return False
        return True
