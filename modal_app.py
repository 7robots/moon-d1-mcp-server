"""Modal.com deployment for moon-d1-mcp selenography server."""

import modal
from pathlib import Path

app = modal.App("moon-d1-mcp")

cloudflare_secret = modal.Secret.from_name("moon-d1-cloudflare")
okta_secret = modal.Secret.from_name("moon-d1-okta")

# Shared volume for OAuth client registration persistence across cold starts.
# All MCP servers can share this volume — each namespaces its own data via app_name.
auth_volume = modal.Volume.from_name("mcp-auth-storage", create_if_missing=True)

# Resolve mcp_deploy_utils source directory (sibling repo)
_mcp_utils_dir = str(Path(__file__).resolve().parent / ".." / "mcp-deploy-utils" / "src" / "mcp_deploy_utils")

# Build image: install deps via uv, bundle all code with copy=True to avoid mount caching
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .run_commands(
        'uv pip install --system "fastmcp>=3.1.0" "httpx>=0.27.0" "pydantic>=2.0.0"'
    )
    .add_local_file("server.py", "/app/server.py", copy=True)
    .add_local_dir("skills", "/app/skills", copy=True)
    .add_local_dir(_mcp_utils_dir, "/app/mcp_deploy_utils", copy=True)
)


@app.function(
    image=image,
    secrets=[cloudflare_secret, okta_secret],
    volumes={"/data": auth_volume},
    timeout=300,
    min_containers=1,
    scaledown_window=1200,
)
@modal.asgi_app()
def web():
    import sys

    sys.path.insert(0, "/app")
    from server import mcp

    return mcp.http_app(transport="streamable-http", stateless_http=True)
