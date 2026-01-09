import asyncio
import json
import os
import random
import time
import traceback
import uuid
from pathlib import Path
from typing import Dict, Optional, List, Any, AsyncGenerator, Tuple

import httpx
import tiktoken
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse

from claude_converter import convert_claude_to_amazonq_request, map_model_name
from claude_stream import ClaudeStreamHandler
from claude_types import ClaudeRequest
from db import init_db, close_db
from http_client import create_client
from openai_converter import (
    convert_openai_messages_to_aq,
    OpenAIStreamState,
    convert_aq_event_to_openai_chunk,
    create_openai_final_chunk,
    convert_aq_response_to_openai_non_streaming,
)
from replicate import send_chat_request
from schemas import (
    Account, AccountCreate, BatchAccountCreate, AccountUpdate,
    AccountListResponse, AccountUsage, AccountUsageListResponse,
    AuthStartBody, AuthStartResponse, AuthStatusDetailResponse, AuthClaimResponse,
    AdminLoginRequest, AdminLoginResponse,
    ChatCompletionRequest,
    FeedAccountsResponse, TokenCountResponse, HealthResponse, DeleteResponse,
)
from token_refresh import refresh_account_token

# ------------------------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------------------------

ENCODING = tiktoken.get_encoding("cl100k_base")
def count_tokens(text: str, apply_multiplier: bool = False) -> int:
    """Counts tokens with tiktoken."""
    if not text or not ENCODING:
        return 0
    token_count = len(ENCODING.encode(text))
    if apply_multiplier:
        token_count = int(token_count * TOKEN_COUNT_MULTIPLIER)
    return token_count

# ------------------------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------------------------
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="v2 OpenAI-compatible Server (Amazon Q Backend)")

# CORS for simple testing in browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Global HTTP Client
# ------------------------------------------------------------------------------

GLOBAL_CLIENT: Optional[httpx.AsyncClient] = None

# ------------------------------------------------------------------------------
# Database helpers
# ------------------------------------------------------------------------------

# Database backend instance (initialized on startup)
_db = None


# ------------------------------------------------------------------------------
# Background token refresh thread
# ------------------------------------------------------------------------------

async def _refresh_stale_tokens():
    while True:
        try:
            await asyncio.sleep(300)  # 5 minutes
            if _db is None:
                print("[Error] Database not initialized, skipping token refresh cycle.")
                continue
            now = time.time()
            
            if LAZY_ACCOUNT_POOL_ENABLED:
                limit = LAZY_ACCOUNT_POOL_SIZE + LAZY_ACCOUNT_POOL_REFRESH_OFFSET
                order_direction = "DESC" if LAZY_ACCOUNT_POOL_ORDER_DESC else "ASC"
                query = f"SELECT id, last_refresh_time FROM accounts WHERE enabled=1 ORDER BY {LAZY_ACCOUNT_POOL_ORDER_BY} {order_direction} LIMIT {limit}"
                rows = await _db.fetchall(query)
            else:
                rows = await _db.fetchall("SELECT id, last_refresh_time FROM accounts WHERE enabled=1")

            for row in rows:
                acc_id, last_refresh = row['id'], row['last_refresh_time']
                should_refresh = False
                if not last_refresh or last_refresh == "never":
                    should_refresh = True
                else:
                    try:
                        last_time = time.mktime(time.strptime(last_refresh, "%Y-%m-%dT%H:%M:%S"))
                        if now - last_time > 1500:  # 25 minutes
                            should_refresh = True
                    except Exception:
                        # Malformed or unparsable timestamp; force refresh
                        should_refresh = True

                if should_refresh:
                    try:
                        acc = await get_account(acc_id)
                        await refresh_account_in_db(acc)
                    except Exception:
                        traceback.print_exc()
                        pass
        except Exception:
            traceback.print_exc()
            pass

# ------------------------------------------------------------------------------
# Env and API Key authorization (keys are independent of AWS accounts)
# ------------------------------------------------------------------------------
def _parse_allowed_keys_env() -> List[str]:
    """
    OPENAI_KEYS is a comma-separated whitelist of API keys for authorization only.
    Example: OPENAI_KEYS="key1,key2,key3"
    - When the list is non-empty, incoming Authorization: Bearer {key} must be one of them.
    - When empty or unset, authorization is effectively disabled (dev mode).
    """
    s = os.getenv("OPENAI_KEYS", "") or ""
    return [x.strip() for x in s.split(",") if x.strip()]

ALLOWED_API_KEYS: List[str] = _parse_allowed_keys_env()
MAX_ERROR_COUNT: int = int(os.getenv("MAX_ERROR_COUNT", "100"))
MAX_RETRY_COUNT: int = int(os.getenv("MAX_RETRY_COUNT", "3"))
TOKEN_COUNT_MULTIPLIER: float = float(os.getenv("TOKEN_COUNT_MULTIPLIER", "1.0"))

# Model mapping: MODEL_MAPPING=源模型:目标模型,源模型2:目标模型2
def _parse_model_mapping() -> Dict[str, str]:
    s = os.getenv("MODEL_MAPPING", "") or ""
    mapping = {}
    for pair in s.split(","):
        if ":" in pair:
            src, dst = pair.split(":", 1)
            if src.strip() and dst.strip():
                mapping[src.strip()] = dst.strip()
    return mapping

MODEL_MAPPING: Dict[str, str] = _parse_model_mapping()

# Lazy Account Pool settings
LAZY_ACCOUNT_POOL_ENABLED: bool = os.getenv("LAZY_ACCOUNT_POOL_ENABLED", "false").lower() in ("true", "1", "yes")
LAZY_ACCOUNT_POOL_SIZE: int = int(os.getenv("LAZY_ACCOUNT_POOL_SIZE", "20"))
LAZY_ACCOUNT_POOL_REFRESH_OFFSET: int = int(os.getenv("LAZY_ACCOUNT_POOL_REFRESH_OFFSET", "10"))
LAZY_ACCOUNT_POOL_ORDER_BY: str = os.getenv("LAZY_ACCOUNT_POOL_ORDER_BY", "created_at")
LAZY_ACCOUNT_POOL_ORDER_DESC: bool = os.getenv("LAZY_ACCOUNT_POOL_ORDER_DESC", "false").lower() in ("true", "1", "yes")

# Validate LAZY_ACCOUNT_POOL_ORDER_BY to prevent SQL injection
if LAZY_ACCOUNT_POOL_ORDER_BY not in ["created_at", "id", "success_count"]:
    LAZY_ACCOUNT_POOL_ORDER_BY = "created_at"

def _is_console_enabled() -> bool:
    """检查是否启用管理控制台"""
    console_env = os.getenv("ENABLE_CONSOLE", "true").strip().lower()
    return console_env not in ("false", "0", "no", "disabled")

CONSOLE_ENABLED: bool = _is_console_enabled()

# Admin authentication configuration
ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "admin")
LOGIN_MAX_ATTEMPTS: int = 5
LOGIN_LOCKOUT_SECONDS: int = 3600  # 1 hour
_login_failures: Dict[str, Dict] = {}  # {ip: {"count": int, "locked_until": float}}

def _extract_bearer(token_header: Optional[str]) -> Optional[str]:
    if not token_header:
        return None
    if token_header.startswith("Bearer "):
        return token_header.split(" ", 1)[1].strip()
    return token_header.strip()

async def _list_enabled_accounts(limit: Optional[int] = None) -> List[Account]:
    if LAZY_ACCOUNT_POOL_ENABLED:
        order_direction = "DESC" if LAZY_ACCOUNT_POOL_ORDER_DESC else "ASC"
        query = f"SELECT * FROM accounts WHERE enabled=1 ORDER BY {LAZY_ACCOUNT_POOL_ORDER_BY} {order_direction}"
        if limit:
            query += f" LIMIT {limit}"
        rows = await _db.fetchall(query)
    else:
        query = "SELECT * FROM accounts WHERE enabled=1 ORDER BY created_at DESC"
        if limit:
            query += f" LIMIT {limit}"
        rows = await _db.fetchall(query)
    return [Account.from_row(r) for r in rows]

async def _list_disabled_accounts() -> List[Account]:
    rows = await _db.fetchall("SELECT * FROM accounts WHERE enabled=0 ORDER BY created_at DESC")
    return [Account.from_row(r) for r in rows]

async def verify_account(account: Account) -> Tuple[bool, Optional[str]]:
    """验证账号可用性"""
    try:
        account = await refresh_account_in_db(account)
        test_request = {
            "conversationState": {
                "currentMessage": {"userInputMessage": {"content": "hello"}},
                "chatTriggerType": "MANUAL"
            }
        }
        _, _, tracker, event_gen = await send_chat_request(
            access_token=account.accessToken,
            messages=[],
            stream=True,
            raw_payload=test_request
        )
        if event_gen:
            async for _ in event_gen:
                break
        return True, None
    except Exception as e:
        if "AccessDenied" in str(e) or "403" in str(e):
            return False, "AccessDenied"
        return False, None

async def resolve_account_for_key(bearer_key: Optional[str], exclude_ids: Optional[List[str]] = None) -> Account:
    """
    Authorize request by OPENAI_KEYS (if configured), then select an AWS account.
    Selection strategy: weighted random based on error rate (lower error rate = higher probability).
    """
    # Authorization: allow admin password to bypass OPENAI_KEYS check (for console testing)
    is_admin = bearer_key and bearer_key == ADMIN_PASSWORD
    if ALLOWED_API_KEYS and not is_admin:
        if not bearer_key or bearer_key not in ALLOWED_API_KEYS:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # Get candidate accounts
    if LAZY_ACCOUNT_POOL_ENABLED:
        candidates = await _list_enabled_accounts(limit=LAZY_ACCOUNT_POOL_SIZE)
    else:
        candidates = await _list_enabled_accounts()

    # Exclude specified accounts (for retry)
    if exclude_ids:
        candidates = [acc for acc in candidates if acc.id not in exclude_ids]

    if not candidates:
        raise HTTPException(status_code=401, detail="No enabled account available")

    # Weighted random selection: lower error rate = higher weight
    def get_weight(acc: Account):
        total = acc.success_count + acc.error_count
        if total == 0:
            return 0.5  # 新账号中等权重
        error_rate = acc.error_count / total
        return max(0.1, 1 - error_rate)  # 最低0.1，保证都有机会

    weights = [get_weight(acc) for acc in candidates]
    return random.choices(candidates, weights=weights, k=1)[0]

# ------------------------------------------------------------------------------
# Token refresh
# ------------------------------------------------------------------------------

async def get_account(account_id: str) -> Account:
    row = await _db.fetchone("SELECT * FROM accounts WHERE id=?", (account_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Account not found")
    return Account.from_row(row)

async def refresh_account_in_db(account: Account) -> Account:
    """刷新账号 token 并更新数据库"""
    try:
        new_access, new_refresh = await refresh_account_token(account, GLOBAL_CLIENT)
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        await _db.execute(
            "UPDATE accounts SET accessToken=?, refreshToken=?, last_refresh_time=?, last_refresh_status=?, updated_at=? WHERE id=?",
            (new_access, new_refresh, now, "success", now, account.id),
        )
        return await get_account(account.id)
    except Exception:
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        await _db.execute(
            "UPDATE accounts SET last_refresh_time=?, last_refresh_status=?, updated_at=? WHERE id=?",
            (now, "failed", now, account.id),
        )
        raise

#todo 待优化错误次数统计 现在是连续错误次数才算
async def _update_stats(account_id: str, success: bool) -> None:
    if success:
        await _db.execute("UPDATE accounts SET success_count=success_count+1, error_count=0, updated_at=? WHERE id=?",
                    (time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), account_id))
    else:
        row = await _db.fetchone("SELECT error_count FROM accounts WHERE id=?", (account_id,))
        if row:
            new_count = (row['error_count'] or 0) + 1
            if new_count >= MAX_ERROR_COUNT:
                await _db.execute("UPDATE accounts SET error_count=?, enabled=0, updated_at=? WHERE id=?",
                           (new_count, time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), account_id))
            else:
                await _db.execute("UPDATE accounts SET error_count=?, updated_at=? WHERE id=?",
                           (new_count, time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), account_id))

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------

async def require_account(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None)
) -> Dict[str, Any]:
    key = _extract_bearer(authorization) if authorization else x_api_key
    return await resolve_account_for_key(key)

def verify_admin_password(authorization: Optional[str] = Header(None)) -> bool:
    """Verify admin password for console access"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={"error": "Unauthorized access", "code": "UNAUTHORIZED"}
        )

    password = authorization[7:]  # Remove "Bearer " prefix

    if password != ADMIN_PASSWORD:
        raise HTTPException(
            status_code=401,
            detail={"error": "Invalid password", "code": "INVALID_PASSWORD"}
        )

    return True



def _sse_format(obj: Dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

# ------------------------------------------------------------------------------
# claude-compatible Chat endpoint
# ------------------------------------------------------------------------------

@app.post("/v1/messages")
async def claude_messages(
    req: ClaudeRequest,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
    x_conversation_id: Optional[str] = Header(default=None, alias="x-conversation-id")
):
    """
    Claude-compatible messages endpoint with retry support.
    """
    # Extract bearer key for authorization
    bearer_key = _extract_bearer(authorization) if authorization else x_api_key

    # 1. Convert request (do this once, before retry loop)
    try:
        aq_request = convert_claude_to_amazonq_request(req, conversation_id=None)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Request conversion failed: {str(e)}")

    # Post-process history to fix message ordering (prevents infinite loops)
    from message_processor import process_claude_history_for_amazonq
    conversation_state = aq_request.get("conversationState", {})
    history = conversation_state.get("history", [])
    if history:
        processed_history = process_claude_history_for_amazonq(history)
        aq_request["conversationState"]["history"] = processed_history

    # Remove duplicate tail userInputMessage that matches currentMessage content
    conversation_state = aq_request.get("conversationState", {})
    current_msg = conversation_state.get("currentMessage", {}).get("userInputMessage", {})
    current_content = (current_msg.get("content") or "").strip()
    history = conversation_state.get("history", [])

    if history and current_content:
        last = history[-1]
        if "userInputMessage" in last:
            last_content = (last["userInputMessage"].get("content") or "").strip()
            if last_content and last_content == current_content:
                history = history[:-1]
                aq_request["conversationState"]["history"] = history
                import logging
                logging.getLogger(__name__).info("Removed duplicate tail userInputMessage to prevent repeated response")

    conversation_state = aq_request.get("conversationState", {})
    conversation_id = conversation_state.get("conversationId")
    response_headers: Dict[str, str] = {}
    if conversation_id:
        response_headers["x-conversation-id"] = conversation_id

    # Calculate input tokens (do this once)
    text_to_count = ""
    if req.system:
        if isinstance(req.system, str):
            text_to_count += req.system
        elif isinstance(req.system, list):
            for item in req.system:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_to_count += item.get("text", "")
    for msg in req.messages:
        if isinstance(msg.content, str):
            text_to_count += msg.content
        elif isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_to_count += item.get("text", "")
    input_tokens = count_tokens(text_to_count, apply_multiplier=True)

    # Retry loop with exponential backoff
    tried_account_ids: List[str] = []
    last_error: Optional[Exception] = None
    max_attempts = MAX_RETRY_COUNT + 1  # +1 for initial attempt

    for attempt in range(max_attempts):
        event_iter = None
        account = None
        try:
            # Get account (excluding previously failed ones)
            # If no accounts available, reset the exclusion list and try again
            try:
                account = await resolve_account_for_key(bearer_key, exclude_ids=tried_account_ids if tried_account_ids else None)
            except HTTPException as he:
                if "No enabled account available" in str(he.detail) and tried_account_ids:
                    # All accounts tried, reset and try again
                    tried_account_ids = []
                    account = await resolve_account_for_key(bearer_key, exclude_ids=None)
                else:
                    raise
            tried_account_ids.append(account.id)

            access = account.accessToken
            if not access:
                refreshed = await refresh_account_in_db(account)
                access = refreshed.accessToken

            # Send request
            _, _, tracker, event_iter = await send_chat_request(
                access_token=access,
                messages=[],
                model=map_model_name(req.model),
                stream=True,
                client=GLOBAL_CLIENT,
                raw_payload=aq_request
            )

            if not event_iter:
                raise Exception("No event stream returned")

            # Try to get the first event to ensure the connection is valid
            first_event = None
            try:
                first_event = await event_iter.__anext__()
            except StopAsyncIteration:
                raise Exception("Empty response from upstream")

            # Success! Create handler and return response
            handler = ClaudeStreamHandler(model=req.model, input_tokens=input_tokens, conversation_id=conversation_id)

            # Capture account for closure
            current_account = account
            current_tracker = tracker

            async def event_generator():
                try:
                    if first_event:
                        event_type, payload = first_event
                        async for sse in handler.handle_event(event_type, payload):
                            yield sse
                    async for event_type, payload in event_iter:
                        async for sse in handler.handle_event(event_type, payload):
                            yield sse
                    async for sse in handler.finish():
                        yield sse
                    await _update_stats(current_account.id, True)
                except GeneratorExit:
                    await _update_stats(current_account.id, current_tracker.has_content if current_tracker else False)
                except Exception:
                    await _update_stats(current_account.id, False)
                    raise

            if req.stream:
                return StreamingResponse(
                    event_generator(),
                    media_type="text/event-stream",
                    headers=response_headers or None
                )
            else:
                # Non-streaming: accumulate response
                content_blocks = []
                usage = {"input_tokens": 0, "output_tokens": 0}
                stop_reason = None
                final_content = []

                async for sse_chunk in event_generator():
                    data_str = None
                    for line in sse_chunk.strip().split('\n'):
                        if line.startswith("data:"):
                            data_str = line[6:].strip()
                            break
                    if not data_str or data_str == "[DONE]":
                        continue
                    try:
                        data = json.loads(data_str)
                        dtype = data.get("type")
                        if dtype == "content_block_start":
                            idx = data.get("index", 0)
                            while len(final_content) <= idx:
                                final_content.append(None)
                            final_content[idx] = data.get("content_block")
                        elif dtype == "content_block_delta":
                            idx = data.get("index", 0)
                            delta = data.get("delta", {})
                            if final_content[idx]:
                                if delta.get("type") == "text_delta":
                                    final_content[idx]["text"] += delta.get("text", "")
                                elif delta.get("type") == "thinking_delta":
                                    final_content[idx].setdefault("thinking", "")
                                    final_content[idx]["thinking"] += delta.get("thinking", "")
                                elif delta.get("type") == "input_json_delta":
                                    if "partial_json" not in final_content[idx]:
                                        final_content[idx]["partial_json"] = ""
                                    final_content[idx]["partial_json"] += delta.get("partial_json", "")
                        elif dtype == "content_block_stop":
                            idx = data.get("index", 0)
                            if final_content[idx] and final_content[idx].get("type") == "tool_use":
                                if "partial_json" in final_content[idx]:
                                    try:
                                        final_content[idx]["input"] = json.loads(final_content[idx]["partial_json"])
                                    except json.JSONDecodeError:
                                        final_content[idx]["input"] = {"error": "invalid json", "partial": final_content[idx]["partial_json"]}
                                    del final_content[idx]["partial_json"]
                        elif dtype == "message_delta":
                            usage = data.get("usage", usage)
                            stop_reason = data.get("delta", {}).get("stop_reason")
                    except (json.JSONDecodeError, Exception):
                        pass

                final_content_cleaned = [c for c in final_content if c is not None]
                for c in final_content_cleaned:
                    c.pop("partial_json", None)

                response_body = {
                    "id": f"msg_{uuid.uuid4()}",
                    "type": "message",
                    "role": "assistant",
                    "model": req.model,
                    "content": final_content_cleaned,
                    "stop_reason": stop_reason,
                    "stop_sequence": None,
                    "usage": usage
                }
                if conversation_id:
                    response_body["conversation_id"] = conversation_id
                    response_body["conversationId"] = conversation_id
                return JSONResponse(content=response_body, headers=response_headers or None)

        except HTTPException:
            # Don't retry on HTTP exceptions (auth errors, etc.)
            if account:
                await _update_stats(account.id, False)
            raise
        except Exception as e:
            # Close event_iter if exists
            if event_iter and hasattr(event_iter, "aclose"):
                try:
                    await event_iter.aclose()
                except Exception:
                    pass
            if account:
                await _update_stats(account.id, False)

            # Check for non-retryable errors
            error_str = str(e)
            if "CONTENT_LENGTH_EXCEEDS_THRESHOLD" in error_str or "Input is too long" in error_str:
                raise HTTPException(status_code=400, detail="Input is too long. Please reduce the context length.")

            last_error = e

            # Log retry attempt and apply exponential backoff
            if attempt < max_attempts - 1:
                import logging
                import asyncio
                # Exponential backoff: 3s, 6s, 12s, 24s... max 30s
                wait_time = min(3 * (2 ** attempt), 30)
                logging.getLogger(__name__).warning(f"Request failed (attempt {attempt + 1}/{max_attempts}), retrying in {wait_time}s with different account: {str(e)}")
                await asyncio.sleep(wait_time)
                continue
            else:
                # All retries exhausted
                raise HTTPException(status_code=502, detail=f"All {max_attempts} attempts failed. Last error: {str(last_error)}")

@app.post("/v1/messages/count_tokens", response_model=TokenCountResponse)
async def count_tokens_endpoint(req: ClaudeRequest) -> TokenCountResponse:
    """
    Count tokens in a message without sending it.
    Compatible with Claude API's /v1/messages/count_tokens endpoint.
    Uses tiktoken for local token counting.
    """
    text_to_count = ""
    
    # Count system prompt tokens
    if req.system:
        if isinstance(req.system, str):
            text_to_count += req.system
        elif isinstance(req.system, list):
            for item in req.system:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_to_count += item.get("text", "")
    
    # Count message tokens
    for msg in req.messages:
        if isinstance(msg.content, str):
            text_to_count += msg.content
        elif isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_to_count += item.get("text", "")
    
    # Count tool definition tokens if present
    if req.tools:
        text_to_count += json.dumps([tool.model_dump() if hasattr(tool, 'model_dump') else tool for tool in req.tools], ensure_ascii=False)
    
    input_tokens = count_tokens(text_to_count, apply_multiplier=True)

    return TokenCountResponse(input_tokens=input_tokens)


# ------------------------------------------------------------------------------
# OpenAI-compatible Chat endpoint
# ------------------------------------------------------------------------------

def _openai_non_streaming_response(
        text: str,
        model: Optional[str],
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
) -> Dict[str, Any]:
    created = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": created,
        "model": model or "unknown",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, account: Account = Depends(require_account)):
    """
    OpenAI-compatible chat endpoint with tool calling support.
    - stream default False
    - supports tools parameter for function calling
    - account is chosen randomly among enabled accounts (API key is for authorization only)
    """
    # Apply model mapping if configured
    request_model = MODEL_MAPPING.get(req.model, req.model) if req.model else req.model
    model = map_model_name(request_model)
    do_stream = bool(req.stream)

    # Calculate prompt tokens
    prompt_text = ""
    for m in req.messages:
        if isinstance(m.content, str):
            prompt_text += m.content
        elif isinstance(m.content, list):
            for item in m.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    prompt_text += item.get("text", "")
    prompt_tokens = count_tokens(prompt_text)

    # Convert messages to AWS Q format
    tools_dict = [t.model_dump() for t in req.tools] if req.tools else None
    messages_dict = [m.model_dump() for m in req.messages]
    aq_request = convert_openai_messages_to_aq(messages_dict, tools=tools_dict, model=model)

    async def _send_upstream() -> Tuple[Any, Any, Any]:
        """发送请求到 AWS Q"""
        access = account.accessToken
        if not access:
            refreshed = await refresh_account_in_db(account)
            access = refreshed.accessToken
            if not access:
                raise HTTPException(status_code=502, detail="Access token unavailable after refresh")

        _, _, tracker, event_iter = await send_chat_request(
            access_token=access,
            messages=[],
            model=model,
            stream=True,
            client=GLOBAL_CLIENT,
            raw_payload=aq_request
        )
        return tracker, event_iter, access

    event_iter = None
    if not do_stream:
        # Non-streaming response
        try:
            tracker, event_iter, _ = await _send_upstream()
            if not event_iter:
                raise Exception("No event stream returned")

            # Collect all events
            events = []
            async for event_type, payload in event_iter:
                events.append((event_type, payload))

            await _update_stats(account.id, True)

            # Convert to OpenAI format
            response = convert_aq_response_to_openai_non_streaming(events, model, prompt_tokens)
            return JSONResponse(content=response)

        except Exception:
            if event_iter and hasattr(event_iter, "aclose"):
                try:
                    await event_iter.aclose()
                except Exception:
                    pass
            await _update_stats(account.id, False)
            raise

    else:
        # Streaming response
        created = int(time.time())
        stream_id = f"chatcmpl-{uuid.uuid4()}"

        try:
            tracker, event_iter, _ = await _send_upstream()
            if not event_iter:
                raise Exception("No event stream returned")

            state = OpenAIStreamState(stream_id, model, created)

            async def event_gen() -> AsyncGenerator[str, None]:
                try:
                    # Send initial role chunk
                    yield _sse_format({
                        "id": stream_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    })
                    state.role_sent = True

                    async for event_type, payload in event_iter:
                        chunk, _ = convert_aq_event_to_openai_chunk(event_type, payload, state)
                        if chunk:
                            # Skip role in delta since we already sent it
                            if "delta" in chunk.get("choices", [{}])[0]:
                                chunk["choices"][0]["delta"].pop("role", None)
                            yield _sse_format(chunk)

                    # Send final chunk
                    final_chunk = create_openai_final_chunk(state, prompt_tokens)
                    yield _sse_format(final_chunk)
                    yield "data: [DONE]\n\n"

                    await _update_stats(account.id, True)

                except GeneratorExit:
                    await _update_stats(account.id, tracker.has_content if tracker else False)
                except Exception:
                    await _update_stats(account.id, False)
                    raise

            return StreamingResponse(event_gen(), media_type="text/event-stream")

        except Exception:
            if event_iter and hasattr(event_iter, "aclose"):
                try:
                    await event_iter.aclose()
                except Exception:
                    pass
            await _update_stats(account.id, False)
            raise

# ------------------------------------------------------------------------------
# Device Authorization (URL Login, 5-minute timeout)
# ------------------------------------------------------------------------------

from auth_flow import register_client_min, device_authorize, poll_token_device_code

# In-memory auth sessions (ephemeral)
AUTH_SESSIONS: Dict[str, Dict[str, Any]] = {}

async def _create_account_from_tokens(
    client_id: str,
    client_secret: str,
    access_token: str,
    refresh_token: Optional[str],
    label: Optional[str],
    enabled: bool,
) -> Account:
    now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    acc_id = str(uuid.uuid4())
    await _db.execute(
        """
        INSERT INTO accounts (id, label, clientId, clientSecret, refreshToken, accessToken, other, last_refresh_time, last_refresh_status, created_at, updated_at, enabled)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            acc_id,
            label,
            client_id,
            client_secret,
            refresh_token,
            access_token,
            None,
            now,
            "success",
            now,
            now,
            1 if enabled else 0,
        ),
    )
    return await get_account(acc_id)

# 管理控制台相关端点 - 仅在启用时注册
if CONSOLE_ENABLED:
    # ------------------------------------------------------------------------------
    # Admin Authentication Endpoints
    # ------------------------------------------------------------------------------

    @app.post("/api/login", response_model=AdminLoginResponse)
    async def admin_login(request: AdminLoginRequest, req: Request) -> AdminLoginResponse:
        """Admin login endpoint - password only, with rate limiting"""
        client_ip = req.client.host if req.client else "unknown"
        now = time.time()

        # Check if locked
        if client_ip in _login_failures:
            info = _login_failures[client_ip]
            if info.get("locked_until", 0) > now:
                remaining = int(info["locked_until"] - now)
                return AdminLoginResponse(
                    success=False,
                    message=f"账号已锁定，请 {remaining // 60} 分钟后重试"
                )

        if request.password == ADMIN_PASSWORD:
            # Clear failures on success
            _login_failures.pop(client_ip, None)
            return AdminLoginResponse(
                success=True,
                message="Login successful"
            )
        else:
            # Track failure
            if client_ip not in _login_failures:
                _login_failures[client_ip] = {"count": 0, "locked_until": 0}
            _login_failures[client_ip]["count"] += 1
            count = _login_failures[client_ip]["count"]

            if count >= LOGIN_MAX_ATTEMPTS:
                _login_failures[client_ip]["locked_until"] = now + LOGIN_LOCKOUT_SECONDS
                return AdminLoginResponse(
                    success=False,
                    message=f"密码错误次数过多，账号已锁定1小时"
                )

            remaining = LOGIN_MAX_ATTEMPTS - count
            return AdminLoginResponse(
                success=False,
                message=f"密码错误，还剩 {remaining} 次尝试机会"
            )

    @app.get("/login", response_class=FileResponse)
    def login_page():
        """Serve the login page"""
        path = BASE_DIR / "frontend" / "login.html"
        if not path.exists():
            raise HTTPException(status_code=404, detail="frontend/login.html not found")
        return FileResponse(str(path))

    # ------------------------------------------------------------------------------
    # Device Authorization Endpoints
    # ------------------------------------------------------------------------------

    @app.post("/v2/auth/start", response_model=AuthStartResponse)
    async def auth_start(body: AuthStartBody, _: bool = Depends(verify_admin_password)) -> AuthStartResponse:
        """
        Start device authorization and return verification URL for user login.
        Session lifetime capped at 5 minutes on claim.
        """
        try:
            cid, csec = await register_client_min()
            dev = await device_authorize(cid, csec)
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"OIDC error: {str(e)}")

        auth_id = str(uuid.uuid4())
        sess = {
            "clientId": cid,
            "clientSecret": csec,
            "deviceCode": dev.get("deviceCode"),
            "interval": int(dev.get("interval", 1)),
            "expiresIn": int(dev.get("expiresIn", 600)),
            "verificationUriComplete": dev.get("verificationUriComplete"),
            "userCode": dev.get("userCode"),
            "startTime": int(time.time()),
            "label": body.label,
            "enabled": True if body.enabled is None else bool(body.enabled),
            "status": "pending",
            "error": None,
            "accountId": None,
        }
        AUTH_SESSIONS[auth_id] = sess
        return AuthStartResponse(
            authId=auth_id,
            verificationUriComplete=sess["verificationUriComplete"],
            userCode=sess["userCode"],
            expiresIn=sess["expiresIn"],
            interval=sess["interval"],
        )

    @app.get("/v2/auth/status/{auth_id}", response_model=AuthStatusDetailResponse)
    async def auth_status(auth_id: str, _: bool = Depends(verify_admin_password)) -> AuthStatusDetailResponse:
        sess = AUTH_SESSIONS.get(auth_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Auth session not found")
        now_ts = int(time.time())
        deadline = sess["startTime"] + min(int(sess.get("expiresIn", 600)), 300)
        remaining = max(0, deadline - now_ts)
        return AuthStatusDetailResponse(
            status=sess.get("status"),
            remaining=remaining,
            error=sess.get("error"),
            accountId=sess.get("accountId"),
        )

    @app.post("/v2/auth/claim/{auth_id}", response_model=AuthClaimResponse)
    async def auth_claim(auth_id: str, _: bool = Depends(verify_admin_password)) -> AuthClaimResponse:
        """
        Block up to 5 minutes to exchange the device code for tokens after user completed login.
        On success, creates an enabled account and returns it.
        """
        sess = AUTH_SESSIONS.get(auth_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Auth session not found")
        if sess.get("status") in ("completed", "timeout", "error"):
            return AuthClaimResponse(
                status=sess["status"],
                accountId=sess.get("accountId"),
                error=sess.get("error"),
            )
        try:
            toks = await poll_token_device_code(
                sess["clientId"],
                sess["clientSecret"],
                sess["deviceCode"],
                sess["interval"],
                sess["expiresIn"],
                max_timeout_sec=300,  # 5 minutes
            )
            access_token = toks.get("accessToken")
            refresh_token = toks.get("refreshToken")
            if not access_token:
                raise HTTPException(status_code=502, detail="No accessToken returned from OIDC")

            acc = await _create_account_from_tokens(
                sess["clientId"],
                sess["clientSecret"],
                access_token,
                refresh_token,
                sess.get("label"),
                sess.get("enabled", True),
            )
            sess["status"] = "completed"
            sess["accountId"] = acc.id
            return AuthClaimResponse(
                status="completed",
                account=acc,
            )
        except TimeoutError:
            sess["status"] = "timeout"
            raise HTTPException(status_code=408, detail="Authorization timeout (5 minutes)")
        except httpx.HTTPError as e:
            sess["status"] = "error"
            sess["error"] = str(e)
            raise HTTPException(status_code=502, detail=f"OIDC error: {str(e)}")

    # ------------------------------------------------------------------------------
    # Accounts Management API
    # ------------------------------------------------------------------------------

    @app.post("/v2/accounts", response_model=Account)
    async def create_account(body: AccountCreate, _: bool = Depends(verify_admin_password)) -> Account:
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        acc_id = str(uuid.uuid4())
        other_str = json.dumps(body.other, ensure_ascii=False) if body.other is not None else None
        enabled_val = 1 if (body.enabled is None or body.enabled) else 0
        await _db.execute(
            """
            INSERT INTO accounts (id, label, clientId, clientSecret, refreshToken, accessToken, other, last_refresh_time, last_refresh_status, created_at, updated_at, enabled)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                acc_id,
                body.label,
                body.clientId,
                body.clientSecret,
                body.refreshToken,
                body.accessToken,
                other_str,
                None,
                "never",
                now,
                now,
                enabled_val,
            ),
        )
        return await get_account(acc_id)


    async def _verify_and_enable_accounts(account_ids: List[str]):
        """后台异步验证并启用账号"""
        for acc_id in account_ids:
            try:
                # 必须先获取完整的账号信息
                account = await get_account(acc_id)
                verify_success, fail_reason = await verify_account(account)
                now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

                if verify_success:
                    await _db.execute("UPDATE accounts SET enabled=1, updated_at=? WHERE id=?", (now, acc_id))
                elif fail_reason:
                    other_dict = json.loads(account.other) if account.other else {}
                    other_dict['failedReason'] = fail_reason
                    await _db.execute("UPDATE accounts SET other=?, updated_at=? WHERE id=?", (json.dumps(other_dict, ensure_ascii=False), now, acc_id))
            except Exception as e:
                print(f"Error verifying account {acc_id}: {e}")
                traceback.print_exc()

    @app.post("/v2/accounts/feed", response_model=FeedAccountsResponse)
    async def create_accounts_feed(request: BatchAccountCreate, _: bool = Depends(verify_admin_password)) -> FeedAccountsResponse:
        """
        统一的投喂接口，接收账号列表，立即存入并后台异步验证。
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        new_account_ids = []

        for i, account_data in enumerate(request.accounts):
            acc_id = str(uuid.uuid4())
            other_dict = account_data.other or {}
            other_dict['source'] = 'feed'
            other_str = json.dumps(other_dict, ensure_ascii=False)

            await _db.execute(
                """
                INSERT INTO accounts (id, label, clientId, clientSecret, refreshToken, accessToken, other, last_refresh_time, last_refresh_status, created_at, updated_at, enabled)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    acc_id,
                    account_data.label or f"批量账号 {i+1}",
                    account_data.clientId,
                    account_data.clientSecret,
                    account_data.refreshToken,
                    account_data.accessToken,
                    other_str,
                    None,
                    "never",
                    now,
                    now,
                    0,  # 初始为禁用状态
                ),
            )
            new_account_ids.append(acc_id)

        # 启动后台任务进行验证，不阻塞当前请求
        if new_account_ids:
            asyncio.create_task(_verify_and_enable_accounts(new_account_ids))

        return FeedAccountsResponse(
            status="processing",
            message=f"{len(new_account_ids)} accounts received and are being verified in the background.",
            account_ids=new_account_ids
        )

    @app.get("/v2/accounts", response_model=AccountListResponse)
    async def list_accounts(_: bool = Depends(verify_admin_password), enabled: Optional[bool] = None, sort_by: str = "created_at", sort_order: str = "desc") -> AccountListResponse:
        query = "SELECT * FROM accounts"
        params = []
        if enabled is not None:
            query += " WHERE enabled=?"
            params.append(1 if enabled else 0)
        sort_field = "created_at" if sort_by not in ["created_at", "success_count"] else sort_by
        order = "DESC" if sort_order.lower() == "desc" else "ASC"
        query += f" ORDER BY {sort_field} {order}"
        rows = await _db.fetchall(query, tuple(params) if params else ())
        accounts = [Account.from_row(r) for r in rows]
        return AccountListResponse(accounts=accounts, count=len(accounts))

    # ------------------------------------------------------------------------------
    # Usage Query
    # ------------------------------------------------------------------------------

    USAGE_LIMITS_URL = "https://q.{region}.amazonaws.com/getUsageLimits"

    async def _get_account_usage(account: Account, client: httpx.AsyncClient) -> Dict[str, Any]:
        """获取单个账号的使用量"""
        result = {"account_id": account.id, "label": account.label or "", "success": False, "usage": None, "error": None}

        access_token = account.accessToken
        if not access_token:
            result["error"] = "无访问令牌"
            return result

        region = "us-east-1"
        url = USAGE_LIMITS_URL.format(region=region)
        params = {"isEmailRequired": "true", "origin": "AI_EDITOR", "resourceType": "AGENTIC_REQUEST"}
        headers = {
            "Authorization": f"Bearer {access_token}",
            "amz-sdk-invocation-id": str(uuid.uuid4()),
            "amz-sdk-request": "attempt=1; max=1",
        }

        try:
            resp = await client.get(url, params=params, headers=headers, timeout=30.0)
            if resp.status_code == 200:
                result["success"] = True
                result["usage"] = resp.json()
            else:
                result["error"] = f"HTTP {resp.status_code}"
        except Exception as e:
            result["error"] = str(e)

        return result

    @app.get("/v2/accounts/usage", response_model=AccountUsageListResponse)
    async def get_all_accounts_usage(_: bool = Depends(verify_admin_password)) -> AccountUsageListResponse:
        """查询所有启用账号的使用量"""
        rows = await _db.fetchall("SELECT * FROM accounts WHERE enabled=1 ORDER BY created_at DESC")
        accounts = [Account.from_row(r) for r in rows]

        if not accounts:
            return AccountUsageListResponse(results=[], total=0, success_count=0)

        results = []
        async with create_client(timeout=30.0) as client:
            tasks = [_get_account_usage(acc, client) for acc in accounts]
            raw_results = await asyncio.gather(*tasks)
            results = [AccountUsage(**r) for r in raw_results]

        success_count = sum(1 for r in results if r.success)
        return AccountUsageListResponse(results=results, total=len(results), success_count=success_count)

    @app.get("/v2/accounts/{account_id}/usage", response_model=AccountUsage)
    async def get_single_account_usage(account_id: str, _: bool = Depends(verify_admin_password)) -> AccountUsage:
        """查询单个账号的使用量"""
        account = await get_account(account_id)
        async with create_client(timeout=30.0) as client:
            result = await _get_account_usage(account, client)
        return AccountUsage(**result)

    @app.get("/v2/accounts/{account_id}", response_model=Account)
    async def get_account_detail(account_id: str, _: bool = Depends(verify_admin_password)) -> Account:
        return await get_account(account_id)

    @app.delete("/v2/accounts/{account_id}", response_model=DeleteResponse)
    async def delete_account(account_id: str, _: bool = Depends(verify_admin_password)) -> DeleteResponse:
        rowcount = await _db.execute("DELETE FROM accounts WHERE id=?", (account_id,))
        if rowcount == 0:
            raise HTTPException(status_code=404, detail="Account not found")
        return DeleteResponse(deleted=account_id)

    @app.patch("/v2/accounts/{account_id}", response_model=Account)
    async def update_account(account_id: str, body: AccountUpdate, _: bool = Depends(verify_admin_password)) -> Account:
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        fields = []
        values: List[Any] = []

        if body.label is not None:
            fields.append("label=?"); values.append(body.label)
        if body.clientId is not None:
            fields.append("clientId=?"); values.append(body.clientId)
        if body.clientSecret is not None:
            fields.append("clientSecret=?"); values.append(body.clientSecret)
        if body.refreshToken is not None:
            fields.append("refreshToken=?"); values.append(body.refreshToken)
        if body.accessToken is not None:
            fields.append("accessToken=?"); values.append(body.accessToken)
        if body.other is not None:
            fields.append("other=?"); values.append(json.dumps(body.other, ensure_ascii=False))
        if body.enabled is not None:
            fields.append("enabled=?"); values.append(1 if body.enabled else 0)

        if not fields:
            return await get_account(account_id)

        fields.append("updated_at=?"); values.append(now)
        values.append(account_id)

        rowcount = await _db.execute(f"UPDATE accounts SET {', '.join(fields)} WHERE id=?", tuple(values))
        if rowcount == 0:
            raise HTTPException(status_code=404, detail="Account not found")
        row = await _db.fetchone("SELECT * FROM accounts WHERE id=?", (account_id,))
        return Account.from_row(row)

    @app.post("/v2/accounts/{account_id}/refresh", response_model=Account)
    async def manual_refresh(account_id: str, _: bool = Depends(verify_admin_password)) -> Account:
        try:
            account = await get_account(account_id)
            return await refresh_account_in_db(account)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Token refresh failed: {str(e)}")

    # ------------------------------------------------------------------------------
    # Simple Frontend (minimal dev test page; full UI in v2/frontend/index.html)
    # ------------------------------------------------------------------------------

    # Frontend inline HTML removed; serving ./frontend/index.html instead (see route below)
    # Note: This route is NOT protected - the HTML file is served freely,
    # but the frontend JavaScript checks authentication and redirects to /login if needed.
    # All API endpoints remain protected.

    @app.get("/", response_class=FileResponse)
    def index():
        path = BASE_DIR / "frontend" / "index.html"
        if not path.exists():
            raise HTTPException(status_code=404, detail="frontend/index.html not found")
        return FileResponse(str(path))

# ------------------------------------------------------------------------------
# Health
# ------------------------------------------------------------------------------

@app.get("/healthz", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")

# ------------------------------------------------------------------------------
# Startup / Shutdown Events
# ------------------------------------------------------------------------------

# async def _verify_disabled_accounts_loop():
#     """后台验证禁用账号任务"""
#     while True:
#         try:
#             await asyncio.sleep(1800)
#             async with _conn() as conn:
#                 accounts = await _list_disabled_accounts(conn)
#                 if accounts:
#                     for account in accounts:
#                         other = account.get('other')
#                         if other:
#                             try:
#                                 other_dict = json.loads(other) if isinstance(other, str) else other
#                                 if other_dict.get('failedReason') == 'AccessDenied':
#                                     continue
#                             except:
#                                 pass
#                         try:
#                             verify_success, fail_reason = await verify_account(account)
#                             now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
#                             if verify_success:
#                                 await conn.execute("UPDATE accounts SET enabled=1, updated_at=? WHERE id=?", (now, account['id']))
#                             elif fail_reason:
#                                 other_dict = {}
#                                 if account.get('other'):
#                                     try:
#                                         other_dict = json.loads(account['other']) if isinstance(account['other'], str) else account['other']
#                                     except:
#                                         pass
#                                 other_dict['failedReason'] = fail_reason
#                                 await conn.execute("UPDATE accounts SET other=?, updated_at=? WHERE id=?", (json.dumps(other_dict, ensure_ascii=False), now, account['id']))
#                             await conn.commit()
#                         except Exception:
#                             pass
#         except Exception:
#             pass

@app.on_event("startup")
async def startup_event():
    """Initialize database and start background tasks on startup."""
    global GLOBAL_CLIENT, _db
    GLOBAL_CLIENT = create_client(
        timeout=1.0,
        read_timeout=300.0,
        connect_timeout=1.0,
        max_connections=60,
        keepalive_expiry=30.0
    )
    _db = await init_db()
    asyncio.create_task(_refresh_stale_tokens())

@app.on_event("shutdown")
async def shutdown_event():
    global GLOBAL_CLIENT
    if GLOBAL_CLIENT:
        await GLOBAL_CLIENT.aclose()
        GLOBAL_CLIENT = None
    await close_db()
