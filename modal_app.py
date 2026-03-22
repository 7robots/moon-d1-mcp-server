"""Modal.com deployment for moon-d1-mcp selenography server."""

import modal

app = modal.App("moon-d1-mcp")

cloudflare_secret = modal.Secret.from_name("moon-d1-cloudflare")
okta_secret = modal.Secret.from_name("moon-d1-okta")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastmcp>=3.1.0", "httpx>=0.27.0", "pydantic>=2.0.0")
    .add_local_file("server.py", "/app/server.py")
    .add_local_dir("skills", "/app/skills")
)


@app.function(image=image, secrets=[cloudflare_secret, okta_secret], timeout=300)
@modal.asgi_app()
def web():
    import sys

    sys.path.insert(0, "/app")

    from server import mcp

    return mcp.http_app(transport="streamable-http", stateless_http=True)


@app.local_entrypoint()
def main():
    print("moon-d1-mcp Modal deployment")
    print()
    print("Commands:")
    print("  modal serve modal_app.py   # local dev with hot-reload")
    print("  modal deploy modal_app.py  # deploy to production")
    print()
    print("Create secrets first:")
    print('  modal secret create moon-d1-cloudflare \\')
    print('    CLOUDFLARE_ID="<your-account-id>" \\')
    print('    CLOUDFLARE_TOKEN="<your-api-token>" \\')
    print('    DATABASE_ID="<your-d1-database-id>"')
    print()
    print('  modal secret create moon-d1-okta \\')
    print('    OKTA_CLIENT_ID="<client-id>" \\')
    print('    OKTA_CLIENT_SECRET="<client-secret>" \\')
    print('    OKTA_DOMAIN="https://your-domain.okta.com" \\')
    print('    OKTA_ISSUER="https://your-domain.okta.com/oauth2/default" \\')
    print('    JWT_SIGNING_KEY="<signing-key>" \\')
    print('    MCP_BASE_URL="https://your-modal-url"')
