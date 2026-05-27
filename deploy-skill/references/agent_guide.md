# Deploying moon-d1-mcp on Modal

Complete workflow for deploying, operating, and rolling back the moon-d1-mcp server on Modal.

## Prerequisites

- `uv` installed (`brew install uv` or equivalent)
- `modal` CLI authenticated (`uv run modal token new` if needed)
- Okta application configured (interactive OAuth + a separate API Services app for M2M bearer)
- Modal secrets `moon-d1-cloudflare` and `moon-d1-okta` populated (see [Secrets](#secrets) below)

## File tour

| File | Role |
|---|---|
| `server.py` | FastMCP server. Builds `auth` and tools. Pure Python; no Modal imports here. |
| `modal_storage.py` | `ModalDictStore` — py-key-value-aio adapter over `modal.Dict`. |
| `modal_app.py` | Modal `App` definition, image build, secrets, ASGI entrypoint. |
| `pyproject.toml` | Dependencies (fastmcp, py-key-value-aio, cryptography, modal). |
| `skills/lunar-selenography/` | SkillProvider data — bundled into the Modal image; served to clients via MCP at runtime. |
| `old-src/` | Archived fastmcp.cloud + Redis implementation, for rollback. |
| `deploy-skill/` | This skill — for Claude Code agents working on this repo. NOT shipped to Modal. |

## Auth wiring

`_create_auth()` in `server.py` returns `MultiAuth(server=OIDCProxy, verifiers=[IntrospectionTokenVerifier])` when `OKTA_CLIENT_SECRET` is present, else `None`. The OIDCProxy is configured with:

- `extra_authorize_params={"scope": "openid profile email offline_access"}` — required for Okta to issue a refresh token.
- `client_storage=_build_client_storage(prefix="moon-d1")` — returns `FernetEncryptionWrapper(PrefixCollectionsWrapper(ModalDictStore(...), prefix="moon-d1"), Fernet(STORAGE_ENCRYPTION_KEY))`.

If `STORAGE_ENCRYPTION_KEY` is unset, `_build_client_storage` returns `None` and OIDCProxy falls back to its default in-memory store. This is the local-dev path.

## Secrets

Two Modal secrets. Refresh values via the Modal dashboard, **not** via the CLI (Modal hides secret values by design).

### `moon-d1-cloudflare`

D1 data-source credentials. These have nothing to do with Cloudflare deployment of this server (that's a separate repo). They're just what the Python server uses to query the D1 database.

| Key | Value |
|---|---|
| `CLOUDFLARE_ID` | Your Cloudflare account ID |
| `CLOUDFLARE_TOKEN` | API token with D1 read access for the database |
| `DATABASE_ID` | UUID of the selenography D1 database |

### `moon-d1-okta`

Auth + signing + storage encryption. The legacy version of this secret (created 2026-03-24) was from the pre-fastmcp.cloud Modal era and may be missing the newer keys. When refreshing, verify ALL of these are present:

| Key | Required | Notes |
|---|---|---|
| `OKTA_CLIENT_ID` | yes | Interactive OAuth client ID |
| `OKTA_CLIENT_SECRET` | yes | Interactive OAuth client secret |
| `OKTA_DOMAIN` | yes | e.g. `https://integrator-9607059.okta.com` |
| `OKTA_ISSUER` | optional | Defaults to `{OKTA_DOMAIN}/oauth2/default` |
| `JWT_SIGNING_KEY` | yes | Fixed string. Without it, Linux containers generate ephemeral keys and tokens invalidate on restart. |
| `STORAGE_ENCRYPTION_KEY` | yes | Fernet key. Generate once and don't lose it (lost key = forced re-auth for all users once). |
| `MCP_BASE_URL` | yes | The Modal-assigned URL of the deployed server, e.g. `https://<workspace>--moon-d1-mcp-web.modal.run`. No `/mcp` suffix. |

Generate a Fernet key:
```bash
uv run python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Generate a JWT signing key:
```bash
uv run python -c "import secrets; print(secrets.token_urlsafe(32))"
```

In Okta, on the OAuth application's Sign On tab → Access Policy, confirm **Refresh Token** is in the allowed grant types. Without it, `offline_access` is silently dropped and users re-auth every ~1 hour.

## Deploy

```bash
# 1. Clean stale bytecode (mandatory — Modal lazy mounts can serve old .pyc)
find . -type d -name __pycache__ -exec rm -rf {} +

# 2. Deploy
uv run modal deploy modal_app.py
```

Output includes the assigned URL. The URL format is `https://<workspace>--moon-d1-mcp-web.modal.run`. **Update `MCP_BASE_URL` in the `moon-d1-okta` secret to match this URL** if it changes — `OIDCProxy` uses `MCP_BASE_URL` for redirect URI validation.

## Verify

### Interactive OAuth (Claude Desktop / Code / Web)

1. Add server URL `https://<workspace>--moon-d1-mcp-web.modal.run/mcp` to your Claude client.
2. Authenticate via Okta in browser.
3. Call a tool (e.g. `moon_get_stats`).
4. Confirm it returns data without errors.

### M2M bearer

```bash
ACCESS_TOKEN=$(curl -sX POST "$OKTA_DOMAIN/oauth2/default/v1/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials" \
  -d "client_id=$SERVICE_APP_CLIENT_ID" \
  -d "client_secret=$SERVICE_APP_CLIENT_SECRET" \
  -d "scope=mcp-access" | jq -r .access_token)

curl -X POST "https://<workspace>--moon-d1-mcp-web.modal.run/mcp" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","method":"tools/list","id":1}'
```

### Cold-start persistence

This is the whole point of the modal.Dict storage. To verify:

1. Authenticate via interactive OAuth.
2. Call a tool, confirm it works.
3. Wait long enough for Modal to scale the container to zero (a few minutes of inactivity is usually enough — you can confirm via `uv run modal app stats moon-d1-mcp`).
4. Call a tool again from the same client.
5. Confirm no re-auth prompt appears. The client uses its FastMCP refresh token, OIDCProxy refreshes the upstream Okta token (which was persisted in modal.Dict), and issues a new FastMCP token transparently.

Symptom of failure: the server log will show `JTI mapping not found (token may have expired)`. That means the modal.Dict isn't being read — most commonly because `STORAGE_ENCRYPTION_KEY` isn't set in the secret, or `OAUTH_DICT_NAME` differs between containers, or the `client_storage` arg was dropped from `OIDCProxy(...)`.

## Operate

### View logs

```bash
uv run modal app logs moon-d1-mcp
```

### Stop the deployment

```bash
uv run modal app stop moon-d1-mcp
```

### Inspect the modal.Dict (debugging only — contents are Fernet-encrypted)

```python
import modal
d = modal.Dict.from_name("moon-d1-oauth-storage")
list(d.keys())  # see what's in there
```

Keys are prefixed with `moon-d1` (the `PrefixCollectionsWrapper` prefix). Values are encrypted blobs — not human-readable.

### Reset all OAuth state (force every user to re-authenticate)

```python
import modal
d = modal.Dict.from_name("moon-d1-oauth-storage")
for k in list(d.keys()):
    d.pop(k)
```

Or rotate `STORAGE_ENCRYPTION_KEY` in the secret — all existing entries become unreadable and the proxy re-authenticates.

## Troubleshooting

### `JTI mapping not found (token may have expired)` after cold start

- Verify `STORAGE_ENCRYPTION_KEY` is set in `moon-d1-okta` secret
- Verify `OAUTH_DICT_NAME` (if set) is the same across deploys
- Confirm `modal_storage.py` was bundled into the image: `uv run modal app describe moon-d1-mcp`

### Users re-auth every ~1 hour even with persistence

- Confirm `extra_authorize_params={"scope": "openid profile email offline_access"}` in OIDCProxy
- In Okta, confirm Refresh Token grant type is allowed in the access policy
- Check Okta System Log for the OIDC grant event — should show a refresh token issued

### Bearer tokens rejected

- Confirm the service app's access policy allows client credentials grants
- Check token hasn't expired (Okta default is 1 day)
- Set `cache_ttl_seconds=0` in `IntrospectionTokenVerifier` to rule out caching

### `Module not found` errors at runtime

- Did you run `find . -type d -name __pycache__ -exec rm -rf {} +` before deploy?
- Confirm the file is listed under `add_local_file` in `modal_app.py`

### Modal secret values are wrong but I can't read them

- Open the Modal dashboard → Secrets → click the secret. The UI lets you view/edit individual key values. CLI doesn't.

## Rollback to fastmcp.cloud

If Modal turns out to be the wrong call:

```bash
git mv old-src/server.py server.py
git mv old-src/pyproject.toml pyproject.toml
git mv old-src/fastmcp.json fastmcp.json
rm modal_app.py modal_storage.py
uv lock
# provision Redis, set REDIS_URL + STORAGE_ENCRYPTION_KEY on fastmcp.cloud
# push to GitHub — fastmcp.cloud auto-deploys
```

The `deploy-skill/` directory should also be removed or updated to point at the fastmcp.cloud pattern.

## Promoting this pattern to `mcp-deploy-utils`

If the Modal pattern works, the next step is to lift it into `~/GitHub/mcp-deploy-utils`:

1. Add `modal_storage.py` to `src/mcp_deploy_utils/` so other servers can import `ModalDictStore` directly.
2. Add a `create_modal_dict_storage(prefix, dict_name)` helper that wraps `ModalDictStore` in `FernetEncryptionWrapper` + `PrefixCollectionsWrapper` (mirrors the inline `_build_client_storage` here).
3. Update `SKILL.md` and `references/agent_guide.md` to describe both the fastmcp.cloud and Modal deployment paths.
4. Add a `modal_app.py` template / generator.

The skill at `~/.claude/skills/mcp-deploy-utils/` would then need a refresh from the repo (it's currently stale by ~2 months even from the fastmcp.cloud version).
