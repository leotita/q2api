import asyncio
import json
import time
from typing import Dict, Tuple, Optional

import httpx

from http_client import create_mounts
from token_refresh import OIDC_BASE, TOKEN_URL, make_headers

# OIDC endpoints
REGISTER_URL = f"{OIDC_BASE}/client/register"
DEVICE_AUTH_URL = f"{OIDC_BASE}/device_authorization"
START_URL = "https://view.awsapps.com/start"


async def post_json(client: httpx.AsyncClient, url: str, payload: Dict) -> httpx.Response:
    payload_str = json.dumps(payload, ensure_ascii=False)
    headers = make_headers()
    return await client.post(url, headers=headers, content=payload_str, timeout=httpx.Timeout(15.0, read=60.0))


async def register_client_min() -> Tuple[str, str]:
    """
    Register an OIDC client (minimal) and return (clientId, clientSecret).
    """
    payload = {
        "clientName": "Amazon Q Developer for command line",
        "clientType": "public",
        "scopes": [
            "codewhisperer:completions",
            "codewhisperer:analysis",
            "codewhisperer:conversations",
        ],
    }
    async with httpx.AsyncClient(mounts=create_mounts()) as client:
        r = await post_json(client, REGISTER_URL, payload)
        r.raise_for_status()
        data = r.json()
        return data["clientId"], data["clientSecret"]


async def device_authorize(client_id: str, client_secret: str) -> Dict:
    """
    Start device authorization. Returns dict that includes:
    - deviceCode
    - interval
    - expiresIn
    - verificationUriComplete
    - userCode
    """
    payload = {
        "clientId": client_id,
        "clientSecret": client_secret,
        "startUrl": START_URL,
    }
    async with httpx.AsyncClient(mounts=create_mounts()) as client:
        r = await post_json(client, DEVICE_AUTH_URL, payload)
        r.raise_for_status()
        return r.json()


async def poll_token_device_code(
    client_id: str,
    client_secret: str,
    device_code: str,
    interval: int,
    expires_in: int,
    max_timeout_sec: Optional[int] = 300,
) -> Dict:
    """
    Poll token with device_code until approved or timeout.
    - Respects upstream expires_in, but caps total time by max_timeout_sec (default 5 minutes).
    Returns token dict with at least 'accessToken' and optionally 'refreshToken'.
    Raises:
      - TimeoutError on timeout
      - httpx.HTTPError for non-recoverable HTTP errors
    """
    payload = {
        "clientId": client_id,
        "clientSecret": client_secret,
        "deviceCode": device_code,
        "grantType": "urn:ietf:params:oauth:grant-type:device_code",
    }

    now = time.time()
    upstream_deadline = now + max(1, int(expires_in))
    cap_deadline = now + max_timeout_sec if (max_timeout_sec and max_timeout_sec > 0) else upstream_deadline
    deadline = min(upstream_deadline, cap_deadline)

    # Ensure interval sane
    poll_interval = max(1, int(interval or 1))

    async with httpx.AsyncClient(mounts=create_mounts()) as client:
        while time.time() < deadline:
            r = await post_json(client, TOKEN_URL, payload)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 400:
                # Expect AuthorizationPendingException early on
                try:
                    err = r.json()
                except Exception:
                    err = {"error": r.text}
                if str(err.get("error")) == "authorization_pending":
                    await asyncio.sleep(poll_interval)
                    continue
                # Other 4xx are errors
                r.raise_for_status()
            # Non-200, non-400
            r.raise_for_status()

    raise TimeoutError("Device authorization expired before approval (timeout reached)")