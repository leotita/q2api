import os
from typing import Optional

import httpx


def get_proxy() -> Optional[str]:
    """获取代理配置"""
    return os.getenv("HTTP_PROXY", "").strip() or None


def create_mounts() -> Optional[dict]:
    """创建代理 mounts 配置"""
    proxy = get_proxy()
    if not proxy:
        return None
    return {
        "https://": httpx.AsyncHTTPTransport(proxy=proxy),
        "http://": httpx.AsyncHTTPTransport(proxy=proxy),
    }


def create_client(
    timeout: float = 60.0,
    read_timeout: Optional[float] = None,
    connect_timeout: Optional[float] = None,
    max_connections: int = 60,
    keepalive_expiry: float = 30.0,
) -> httpx.AsyncClient:
    """
    创建配置好的 AsyncClient

    Args:
        timeout: 默认超时时间
        read_timeout: 读取超时（流式响应需要更长时间）
        connect_timeout: 连接超时
        max_connections: 最大连接数
        keepalive_expiry: 连接保持时间
    """
    mounts = create_mounts()
    limits = httpx.Limits(
        max_keepalive_connections=max_connections,
        max_connections=max_connections,
        keepalive_expiry=keepalive_expiry
    )
    timeout_config = httpx.Timeout(
        connect=connect_timeout or timeout,
        read=read_timeout or timeout,
        write=timeout,
        pool=timeout
    )
    return httpx.AsyncClient(mounts=mounts, timeout=timeout_config, limits=limits)
