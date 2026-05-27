---
name: moon-d1-modal-deploy
description: "Deploy and operate moon-d1-mcp on Modal with Okta OAuth + bearer auth and modal.Dict-backed OAuth state. Use when adding/modifying auth, redeploying to Modal, rotating secrets, or debugging OAuth cold-start re-auth. Triggers: deploying moon-d1-mcp, Modal deployment, modal.Dict OAuth storage, FastMCP on Modal, Okta auth for moon-d1, refresh-token cold-start fix, mcp-deploy-utils Modal pivot."
---

# moon-d1-modal-deploy

Experimental Modal-native deployment pattern for FastMCP servers. First implemented in `moon-d1-mcp-server`. If validated here, this pattern will be lifted into `~/GitHub/mcp-deploy-utils` for broader use.

## Why Modal instead of fastmcp.cloud

fastmcp.cloud was simple but had a structural problem: containers scale to zero on idle, and `OIDCProxy`'s in-memory `client_storage` wipes OAuth state on every cold start. The fastmcp.cloud fix required provisioning a separate Redis service (Redis Cloud free tier) to back the storage — extra infra, extra secret to rotate, encryption-at-rest wrapped manually.

Modal offers a native distributed KV (`modal.Dict`) that solves the cold-start problem without any external dependency. Trade-offs:

| | Modal | fastmcp.cloud + Redis |
|---|---|---|
| Storage backend | `modal.Dict` (native) | Redis Cloud (external) |
| Extra service to provision | None | Redis instance |
| Storage TTL | 7d inactivity per entry | Configurable |
| Deployment | `modal deploy modal_app.py` | git push (auto-deploy) |
| Cold-start re-auth | Solved by `modal.Dict` | Solved by Redis |
| Container startup latency | ~few seconds | similar |

The fastmcp.cloud archive lives in `old-src/` so we can revert if Modal turns out to be the wrong choice.

## Layout

```
moon-d1-mcp-server/
├── server.py                 # FastMCP server + auth wiring
├── modal_storage.py          # ModalDictStore (py-key-value-aio adapter)
├── modal_app.py              # Modal deployment config
├── pyproject.toml            # deps: fastmcp, py-key-value-aio, cryptography, modal
├── skills/
│   └── lunar-selenography/   # SkillProvider data — ships INSIDE the Modal image
├── deploy-skill/             # this skill (dev-facing, not deployed)
│   ├── SKILL.md
│   └── references/
│       └── agent_guide.md
└── old-src/                  # fastmcp.cloud + Redis archive
```

## Architecture

**OAuth state persistence:**

```
OIDCProxy.client_storage
  └── FernetEncryptionWrapper            # encryption-at-rest (STORAGE_ENCRYPTION_KEY)
        └── PrefixCollectionsWrapper     # namespacing (prefix="moon-d1")
              └── ModalDictStore         # bottom layer
                    └── modal.Dict.from_name("moon-d1-oauth-storage", create_if_missing=True)
```

The wrappers (Fernet + prefix) are unchanged from the fastmcp.cloud era — only the bottom store swapped from `RedisStore` to `ModalDictStore`. Encryption-at-rest still matters even with Modal's managed storage.

**Auth:**

`MultiAuth(server=OIDCProxy(...), verifiers=[IntrospectionTokenVerifier(...)])` — accepts both interactive Okta OAuth (Claude clients) and Okta-issued bearer tokens (M2M, agents). Same shape as the fastmcp.cloud version.

## Modal-specific gotchas

These are the patterns to get right or things will fail in confusing ways:

- **`mcp_deploy_utils` is NOT used.** That package is for fastmcp.cloud inlining. On Modal we have direct module access — we can write our own `modal_storage.py` and import it cleanly.
- **Import order inside `web()`.** The `from server import mcp` happens inside the function body, not at the top of `modal_app.py`. Modal injects secrets before the function body runs but after `modal_app.py` is imported. Doing it inside the function means env vars are set when `FastMCP(..., auth=_create_auth())` evaluates at module load.
- **`copy=True`** on every `add_local_file`/`add_local_dir`. Without it, Modal uses lazy mounts and can serve stale `__pycache__` bytecode.
- **Pre-deploy `__pycache__` cleanup.** Run `find . -type d -name __pycache__ -exec rm -rf {} +` before every `modal deploy`. Stale `.pyc` from a different Python version has caused production bugs before.
- **`MCP_BASE_URL`** is the root Modal URL (`https://<workspace>--moon-d1-mcp-web.modal.run`). No `/mcp` suffix.
- **`OAUTH_DICT_NAME`** env var lets you override the Dict name. Defaults to `moon-d1-oauth-storage`. Setting a per-environment value (e.g. `moon-d1-oauth-storage-staging`) gives clean isolation between environments without code change.
- **`modal.Dict` 7-day inactivity TTL.** Entries auto-expire after 7 days with no reads/writes. Active users (refresh tokens used at least weekly) are unaffected. Users idle >7d re-authenticate once.

## Modal secrets

| Secret | Keys |
|---|---|
| `moon-d1-cloudflare` | `CLOUDFLARE_ID`, `CLOUDFLARE_TOKEN`, `DATABASE_ID` |
| `moon-d1-okta` | `OKTA_CLIENT_ID`, `OKTA_CLIENT_SECRET`, `OKTA_DOMAIN`, `OKTA_ISSUER` (opt), `JWT_SIGNING_KEY`, `STORAGE_ENCRYPTION_KEY`, `MCP_BASE_URL` |

Both names match the legacy pre-fastmcp.cloud Modal secrets in the workspace. The keys inside `moon-d1-okta` may need refreshing — at minimum confirm `STORAGE_ENCRYPTION_KEY` and `MCP_BASE_URL` are present (those weren't in the older Modal-volume pattern).

Generate a fresh Fernet key:
```bash
uv run python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

## Deploy

```bash
find . -type d -name __pycache__ -exec rm -rf {} +
uv run modal deploy modal_app.py
```

## Detailed guides

- **Step-by-step guide**: [references/agent_guide.md](references/agent_guide.md) — full workflow including local dev, troubleshooting, and rollback to fastmcp.cloud.

## Reference

- `~/GitHub/mcp-deploy-utils/` — the current Cloudflare/fastmcp.cloud reference (out of date for the Modal pattern; will be updated after this experiment).
- `old-src/` (in this repo) — the previous fastmcp.cloud + Redis implementation, kept for rollback.
