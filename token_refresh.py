import time
import uuid
from typing import Dict, Optional, Tuple

import httpx

from http_client import create_client
from schemas import Account


# ------------------------------------------------------------------------------
# OIDC Constants
# ------------------------------------------------------------------------------

OIDC_BASE = "https://oidc.us-east-1.amazonaws.com"
TOKEN_URL = f"{OIDC_BASE}/token"

USER_AGENT = "aws-sdk-rust/1.3.9 os/windows lang/rust/1.87.0"
X_AMZ_USER_AGENT = "aws-sdk-rust/1.3.9 ua/2.1 api/ssooidc/1.88.0 os/windows lang/rust/1.87.0 m/E app/AmazonQ-For-CLI"
AMZ_SDK_REQUEST = "attempt=1; max=3"


def make_headers() -> Dict[str, str]:
    """生成 OIDC 请求头"""
    return {
        "content-type": "application/json",
        "user-agent": USER_AGENT,
        "x-amz-user-agent": X_AMZ_USER_AGENT,
        "amz-sdk-request": AMZ_SDK_REQUEST,
        "amz-sdk-invocation-id": str(uuid.uuid4()),
    }


async def refresh_token(
    client_id: str,
    client_secret: str,
    refresh_token_value: str,
    client: Optional[httpx.AsyncClient] = None,
) -> Dict[str, str]:
    """
    使用 refresh_token 获取新的 access_token。

    Returns:
        包含 accessToken 和可能的新 refreshToken 的字典

    Raises:
        httpx.HTTPError: 请求失败时抛出
    """
    payload = {
        "grantType": "refresh_token",
        "clientId": client_id,
        "clientSecret": client_secret,
        "refreshToken": refresh_token_value,
    }

    if client:
        r = await client.post(TOKEN_URL, headers=make_headers(), json=payload)
        r.raise_for_status()
        return r.json()
    else:
        async with create_client(timeout=60.0) as temp_client:
            r = await temp_client.post(TOKEN_URL, headers=make_headers(), json=payload)
            r.raise_for_status()
            return r.json()


async def refresh_account_token(
    account: Account,
    client: Optional[httpx.AsyncClient] = None,
) -> Tuple[str, str]:
    """
    刷新账号的 access_token。

    Args:
        account: 账号对象
        client: 可选的 HTTP 客户端

    Returns:
        (new_access_token, new_refresh_token) 元组

    Raises:
        ValueError: 账号缺少必要字段
        httpx.HTTPError: Token 刷新失败
    """
    if not account.clientId or not account.clientSecret or not account.refreshToken:
        raise ValueError("Account missing clientId/clientSecret/refreshToken for refresh")

    data = await refresh_token(
        client_id=account.clientId,
        client_secret=account.clientSecret,
        refresh_token_value=account.refreshToken,
        client=client
    )
    new_access = data.get("accessToken", "")
    new_refresh = data.get("refreshToken", account.refreshToken)
    return new_access, new_refresh
