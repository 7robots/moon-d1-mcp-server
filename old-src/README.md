# old-src — fastmcp.cloud archive

Snapshot of the server's pre-pivot configuration, when it was deployed to fastmcp.cloud with Redis-backed OAuth client_storage. Kept in case we need to fall back to fastmcp.cloud while the Modal pattern is being experimented with.

## Contents

- `server.py` — FastMCP server with inline `_build_client_storage()` using `RedisStore` + `FernetEncryptionWrapper` + `PrefixCollectionsWrapper`.
- `pyproject.toml` — deps include `py-key-value-aio[redis]` and `cryptography`.
- `fastmcp.json` — fastmcp.cloud deployment config (transport, env var mapping).

## To restore the fastmcp.cloud deployment

1. Move these three files back to the repo root (`git mv old-src/server.py server.py` etc.).
2. Re-provision Redis + set `REDIS_URL` and `STORAGE_ENCRYPTION_KEY` on fastmcp.cloud.
3. Push to GitHub — fastmcp.cloud auto-deploys.

## Why the pivot

Experimenting with Modal-native deployment using `modal.Dict` for OAuth state instead of Redis. See the repo root's `deploy-skill/` for the new pattern.
