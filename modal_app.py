"""Modal deployment for moon-d1-mcp.

OAuth state lives in a named modal.Dict (see modal_storage.py) so refresh
tokens survive container cold starts. Secrets are split: moon-d1-cloudflare
holds D1 data-source credentials, moon-d1-okta holds auth + JWT signing +
storage encryption keys.
"""

from pathlib import Path

import modal

app = modal.App("moon-d1-mcp")

secrets = [
    modal.Secret.from_name("moon-d1-cloudflare"),
    modal.Secret.from_name("moon-d1-okta"),
]

_repo_root = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject(str(_repo_root / "pyproject.toml"))
    .add_local_file(str(_repo_root / "server.py"), "/app/server.py", copy=True)
    .add_local_file(str(_repo_root / "modal_storage.py"), "/app/modal_storage.py", copy=True)
    .add_local_dir(str(_repo_root / "skills"), "/app/skills", copy=True)
)


@app.function(image=image, secrets=secrets, timeout=300)
@modal.asgi_app()
def web():
    import sys

    sys.path.insert(0, "/app")
    from server import mcp

    return mcp.http_app(transport="streamable-http", stateless_http=True)
