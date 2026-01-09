from typing import Optional, List, Dict, Any

from pydantic import BaseModel


# ------------------------------------------------------------------------------
# Account Models
# ------------------------------------------------------------------------------

class Account(BaseModel):
    """数据库 accounts 表的完整模型"""
    id: str
    label: Optional[str] = None
    clientId: Optional[str] = None
    clientSecret: Optional[str] = None
    refreshToken: Optional[str] = None
    accessToken: Optional[str] = None
    other: Optional[str] = None
    last_refresh_time: Optional[str] = None
    last_refresh_status: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    enabled: bool = True
    error_count: int = 0
    success_count: int = 0

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Account":
        """从数据库行构造 Account"""
        if row is None:
            raise ValueError("Row cannot be None")
        d = dict(row)
        # normalize enabled to bool
        if "enabled" in d and d["enabled"] is not None:
            d["enabled"] = bool(int(d["enabled"])) if isinstance(d["enabled"], (int, str)) else bool(d["enabled"])
        return cls(**d)


class AccountCreate(BaseModel):
    label: Optional[str] = None
    clientId: str
    clientSecret: str
    refreshToken: Optional[str] = None
    accessToken: Optional[str] = None
    other: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = True


class BatchAccountCreate(BaseModel):
    accounts: List[AccountCreate]


class AccountUpdate(BaseModel):
    label: Optional[str] = None
    clientId: Optional[str] = None
    clientSecret: Optional[str] = None
    refreshToken: Optional[str] = None
    accessToken: Optional[str] = None
    other: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = None


class AccountListResponse(BaseModel):
    accounts: List[Account]
    count: int


class AccountUsage(BaseModel):
    account_id: str
    label: str
    success: bool
    usage: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class AccountUsageListResponse(BaseModel):
    results: List[AccountUsage]
    total: int
    success_count: int


# ------------------------------------------------------------------------------
# Auth Models
# ------------------------------------------------------------------------------

class AuthStartBody(BaseModel):
    label: Optional[str] = None
    enabled: Optional[bool] = True


class AuthStartResponse(BaseModel):
    authId: str
    verificationUriComplete: str
    userCode: str
    expiresIn: int
    interval: int


class AuthStatusDetailResponse(BaseModel):
    status: str
    remaining: int
    error: Optional[str] = None
    accountId: Optional[str] = None


class AuthClaimResponse(BaseModel):
    status: str
    account: Optional[Account] = None
    accountId: Optional[str] = None
    error: Optional[str] = None


class AdminLoginRequest(BaseModel):
    password: str


class AdminLoginResponse(BaseModel):
    success: bool
    message: str


# ------------------------------------------------------------------------------
# Chat Models (OpenAI Compatible)
# ------------------------------------------------------------------------------

class OpenAIFunctionDef(BaseModel):
    """OpenAI function definition"""
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class OpenAIToolDef(BaseModel):
    """OpenAI tool definition"""
    type: str = "function"
    function: OpenAIFunctionDef


class OpenAIToolCallFunction(BaseModel):
    """OpenAI tool call function info"""
    name: str
    arguments: str


class OpenAIToolCall(BaseModel):
    """OpenAI tool call in assistant message"""
    id: str
    type: str = "function"
    function: OpenAIToolCallFunction


class ChatMessage(BaseModel):
    """OpenAI chat message format"""
    role: str
    content: Optional[Any] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None
    tool_call_id: Optional[str] = None  # For tool response messages
    name: Optional[str] = None  # Tool name for tool response


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request"""
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    tools: Optional[List[OpenAIToolDef]] = None
    tool_choice: Optional[Any] = None  # "auto", "none", or {"type": "function", "function": {"name": "..."}}
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


# ------------------------------------------------------------------------------
# Common Response Models
# ------------------------------------------------------------------------------

class FeedAccountsResponse(BaseModel):
    status: str
    message: str
    account_ids: List[str]


class TokenCountResponse(BaseModel):
    input_tokens: int


class HealthResponse(BaseModel):
    status: str


class DeleteResponse(BaseModel):
    deleted: str
