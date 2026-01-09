"""
OpenAI 格式与 AWS Q 格式的双向转换器

主要功能:
1. convert_openai_tools_to_aq: OpenAI tools 定义转换为 AWS Q 格式
2. convert_openai_messages_to_aq: OpenAI messages 转换为 AWS Q 请求
3. convert_aq_response_to_openai: AWS Q 响应转换为 OpenAI 格式
"""
import json
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Pydantic Models for OpenAI Format
# ------------------------------------------------------------------------------

class OpenAIFunction(BaseModel):
    """OpenAI function definition"""
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class OpenAITool(BaseModel):
    """OpenAI tool definition"""
    type: str = "function"
    function: OpenAIFunction


class OpenAIToolCall(BaseModel):
    """OpenAI tool call in assistant message"""
    id: str
    type: str = "function"
    function: Dict[str, str]  # {name, arguments}


class OpenAIMessage(BaseModel):
    """OpenAI message format"""
    role: str
    content: Optional[Any] = None
    tool_calls: Optional[List[OpenAIToolCall]] = None
    tool_call_id: Optional[str] = None  # For tool response messages
    name: Optional[str] = None  # Tool name for tool response


# ------------------------------------------------------------------------------
# Tool Definition Conversion: OpenAI -> AWS Q
# ------------------------------------------------------------------------------

def convert_openai_tool_to_aq(tool: Dict[str, Any]) -> Dict[str, Any]:
    """
    将单个 OpenAI tool 定义转换为 AWS Q 格式

    OpenAI 格式:
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather info",
            "parameters": {"type": "object", "properties": {...}}
        }
    }

    AWS Q 格式:
    {
        "toolSpecification": {
            "name": "get_weather",
            "description": "Get weather info",
            "inputSchema": {"json": {"type": "object", "properties": {...}}}
        }
    }
    """
    func = tool.get("function", {})
    name = func.get("name", "")
    description = func.get("description", "") or ""
    parameters = func.get("parameters", {"type": "object", "properties": {}})

    # AWS Q 限制 description 长度
    if len(description) > 10240:
        description = description[:10100] + "\n\n...(truncated)"

    return {
        "toolSpecification": {
            "name": name,
            "description": description,
            "inputSchema": {"json": parameters}
        }
    }


def convert_openai_tools_to_aq(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将 OpenAI tools 列表转换为 AWS Q 格式

    Args:
        tools: OpenAI 格式的工具定义列表

    Returns:
        AWS Q 格式的工具定义列表
    """
    if not tools:
        return []
    return [convert_openai_tool_to_aq(t) for t in tools]


# ------------------------------------------------------------------------------
# Message Conversion: OpenAI -> AWS Q
# ------------------------------------------------------------------------------

def _get_current_timestamp() -> str:
    """获取当前时间戳"""
    now = datetime.now().astimezone()
    weekday = now.strftime("%A")
    iso_time = now.isoformat(timespec='milliseconds')
    return f"{weekday}, {iso_time}"


def _extract_text_content(content: Any) -> str:
    """从 OpenAI content 中提取文本"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def _convert_tool_result_to_aq(tool_call_id: str, content: Any, is_error: bool = False) -> Dict[str, Any]:
    """
    将 OpenAI tool response 转换为 AWS Q toolResult 格式

    Args:
        tool_call_id: 工具调用 ID
        content: 工具执行结果内容
        is_error: 是否为错误结果

    Returns:
        AWS Q toolResult 格式
    """
    text_content = _extract_text_content(content)
    if not text_content.strip():
        text_content = "Command executed successfully" if not is_error else "Tool execution failed"

    return {
        "toolUseId": tool_call_id,
        "content": [{"text": text_content}],
        "status": "error" if is_error else "success"
    }


def convert_openai_messages_to_aq(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    将 OpenAI messages 转换为 AWS Q 请求格式

    策略:
    1. 提取最后一条 user 消息作为 currentMessage
    2. 之前的消息转换为 history
    3. tool response 消息转换为 toolResults

    Args:
        messages: OpenAI 格式的消息列表
        tools: OpenAI 格式的工具定义列表
        model: 模型名称

    Returns:
        AWS Q 请求体
    """
    conversation_id = str(uuid.uuid4())

    # 转换工具定义
    aq_tools = convert_openai_tools_to_aq(tools) if tools else []

    # 分离历史消息和当前消息
    history = []
    current_content = ""
    current_tool_results = []

    # 处理消息，构建历史和当前消息
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "")
        content = msg.get("content")

        if role == "system":
            # System prompt 会被添加到 currentMessage 的 content 前面
            i += 1
            continue

        elif role == "user":
            # 检查是否是最后一条 user 消息
            is_last_user = True
            for j in range(i + 1, len(messages)):
                if messages[j].get("role") == "user":
                    is_last_user = False
                    break

            if is_last_user:
                # 这是最后一条 user 消息，作为 currentMessage
                current_content = _extract_text_content(content)
            else:
                # 添加到历史
                history.append({
                    "userInputMessage": {
                        "content": _extract_text_content(content),
                        "userInputMessageContext": {},
                        "origin": "KIRO_CLI"
                    }
                })
            i += 1

        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            text_content = _extract_text_content(content)

            assistant_msg = {
                "assistantResponseMessage": {
                    "content": text_content
                }
            }

            # 处理工具调用
            if tool_calls:
                tool_uses = []
                for tc in tool_calls:
                    tc_id = tc.get("id", str(uuid.uuid4()))
                    func = tc.get("function", {})
                    tool_uses.append({
                        "toolUseId": tc_id,
                        "name": func.get("name", ""),
                        "input": json.loads(func.get("arguments", "{}")) if isinstance(func.get("arguments"), str) else func.get("arguments", {})
                    })
                if tool_uses:
                    assistant_msg["assistantResponseMessage"]["toolUses"] = tool_uses

            history.append(assistant_msg)
            i += 1

        elif role == "tool":
            # Tool response 消息
            tool_call_id = msg.get("tool_call_id", "")
            tool_result = _convert_tool_result_to_aq(tool_call_id, content)

            # 检查这个 tool response 是否属于最后一轮对话
            # 如果后面还有 user 消息，则添加到历史中对应的 user 消息
            has_user_after = False
            for j in range(i + 1, len(messages)):
                if messages[j].get("role") == "user":
                    has_user_after = True
                    break

            if has_user_after:
                # 添加到历史中的下一个 user 消息
                # 先收集所有连续的 tool responses
                tool_results_batch = [tool_result]
                i += 1
                while i < len(messages) and messages[i].get("role") == "tool":
                    tr = _convert_tool_result_to_aq(
                        messages[i].get("tool_call_id", ""),
                        messages[i].get("content")
                    )
                    tool_results_batch.append(tr)
                    i += 1

                # 创建一个带 toolResults 的 user 消息
                # content 不能为空，使用占位符
                history.append({
                    "userInputMessage": {
                        "content": "[Tool results]",
                        "userInputMessageContext": {
                            "toolResults": tool_results_batch
                        },
                        "origin": "KIRO_CLI"
                    }
                })
            else:
                # 属于当前消息的 tool results
                current_tool_results.append(tool_result)
                i += 1
        else:
            i += 1

    # 构建 system prompt
    system_text = ""
    for msg in messages:
        if msg.get("role") == "system":
            system_text += _extract_text_content(msg.get("content", "")) + "\n"

    # 构建 currentMessage content
    formatted_content = ""
    if current_content or not current_tool_results:
        formatted_content = (
            "--- CONTEXT ENTRY BEGIN ---\n"
            f"Current time: {_get_current_timestamp()}\n"
            "--- CONTEXT ENTRY END ---\n\n"
            "--- USER MESSAGE BEGIN ---\n"
            f"{current_content}\n"
            "--- USER MESSAGE END ---"
        )

    if system_text.strip():
        formatted_content = (
            "--- SYSTEM PROMPT BEGIN ---\n"
            f"{system_text.strip()}\n"
            "--- SYSTEM PROMPT END ---\n\n"
            f"{formatted_content}"
        )

    # 构建 userInputMessageContext
    user_ctx: Dict[str, Any] = {
        "envState": {
            "operatingSystem": "macos",
            "currentWorkingDirectory": "/"
        }
    }

    if aq_tools:
        user_ctx["tools"] = aq_tools

    if current_tool_results:
        user_ctx["toolResults"] = current_tool_results

    # 构建最终请求
    user_input_msg: Dict[str, Any] = {
        "content": formatted_content,
        "userInputMessageContext": user_ctx,
        "origin": "KIRO_CLI"
    }

    if model:
        user_input_msg["modelId"] = model

    return {
        "conversationState": {
            "conversationId": conversation_id,
            "history": history,
            "currentMessage": {
                "userInputMessage": user_input_msg
            },
            "chatTriggerType": "MANUAL"
        }
    }


# ------------------------------------------------------------------------------
# Response Conversion: AWS Q -> OpenAI
# ------------------------------------------------------------------------------

class OpenAIStreamState:
    """
    管理 OpenAI 流式响应的状态

    用于跟踪:
    - 累积的文本内容
    - 工具调用信息
    - 是否已发送角色信息
    """
    def __init__(self, stream_id: str, model: str, created: int):
        self.stream_id = stream_id
        self.model = model
        self.created = created
        self.content = ""
        self.tool_calls: List[Dict[str, Any]] = []
        self.current_tool_index = 0
        self.role_sent = False
        self.finish_reason: Optional[str] = None

    def add_content(self, text: str) -> None:
        """添加文本内容"""
        self.content += text

    def add_tool_call(self, tool_use_id: str, name: str, arguments: str) -> int:
        """
        添加工具调用

        Returns:
            工具调用的索引
        """
        index = len(self.tool_calls)
        self.tool_calls.append({
            "index": index,
            "id": tool_use_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments
            }
        })
        return index


def convert_aq_event_to_openai_chunk(
    event_type: str,
    payload: Dict[str, Any],
    state: OpenAIStreamState
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    将单个 AWS Q 事件转换为 OpenAI 流式响应块

    Args:
        event_type: AWS Q 事件类型
        payload: 事件数据
        state: 流式响应状态

    Returns:
        (OpenAI chunk dict, is_done)
    """
    chunk = None
    is_done = False

    # 处理文本响应
    if event_type == "assistantResponseEvent":
        content = payload.get("content", "")
        if content:
            state.add_content(content)

            delta: Dict[str, Any] = {"content": content}
            if not state.role_sent:
                delta["role"] = "assistant"
                state.role_sent = True

            chunk = {
                "id": state.stream_id,
                "object": "chat.completion.chunk",
                "created": state.created,
                "model": state.model,
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": None
                }]
            }

    # 处理工具调用
    elif event_type == "toolUseEvent":
        tool_use_id = payload.get("toolUseId", str(uuid.uuid4()))
        name = payload.get("name", "")
        input_data = payload.get("input", {})
        arguments = json.dumps(input_data, ensure_ascii=False) if isinstance(input_data, dict) else str(input_data)

        index = state.add_tool_call(tool_use_id, name, arguments)

        # 发送工具调用开始
        delta: Dict[str, Any] = {
            "tool_calls": [{
                "index": index,
                "id": tool_use_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments
                }
            }]
        }
        if not state.role_sent:
            delta["role"] = "assistant"
            state.role_sent = True

        chunk = {
            "id": state.stream_id,
            "object": "chat.completion.chunk",
            "created": state.created,
            "model": state.model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": None
            }]
        }

    # 处理结束事件
    elif event_type in ("messageMetadataEvent", "supplementaryWebLinksEvent"):
        # 这些事件表示响应即将结束
        pass

    return chunk, is_done


def create_openai_final_chunk(state: OpenAIStreamState, prompt_tokens: int = 0) -> Dict[str, Any]:
    """
    创建 OpenAI 流式响应的最终块

    Args:
        state: 流式响应状态
        prompt_tokens: 输入 token 数

    Returns:
        最终的 OpenAI chunk
    """
    # 确定 finish_reason
    finish_reason = "tool_calls" if state.tool_calls else "stop"

    # 计算 completion tokens (简单估算)
    completion_tokens = len(state.content) // 4 + len(json.dumps(state.tool_calls)) // 4 if state.tool_calls else len(state.content) // 4

    return {
        "id": state.stream_id,
        "object": "chat.completion.chunk",
        "created": state.created,
        "model": state.model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }


def convert_aq_response_to_openai_non_streaming(
    events: List[Tuple[str, Dict[str, Any]]],
    model: str,
    prompt_tokens: int = 0
) -> Dict[str, Any]:
    """
    将 AWS Q 事件列表转换为 OpenAI 非流式响应

    Args:
        events: AWS Q 事件列表 [(event_type, payload), ...]
        model: 模型名称
        prompt_tokens: 输入 token 数

    Returns:
        OpenAI 格式的完整响应
    """
    import time

    content = ""
    tool_calls = []

    for event_type, payload in events:
        if event_type == "assistantResponseEvent":
            content += payload.get("content", "")
        elif event_type == "toolUseEvent":
            tool_use_id = payload.get("toolUseId", str(uuid.uuid4()))
            name = payload.get("name", "")
            input_data = payload.get("input", {})
            arguments = json.dumps(input_data, ensure_ascii=False) if isinstance(input_data, dict) else str(input_data)

            tool_calls.append({
                "id": tool_use_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments
                }
            })

    # 构建 message
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": content if content else None
    }

    if tool_calls:
        message["tool_calls"] = tool_calls

    # 确定 finish_reason
    finish_reason = "tool_calls" if tool_calls else "stop"

    # 计算 tokens
    completion_tokens = len(content) // 4 + (len(json.dumps(tool_calls)) // 4 if tool_calls else 0)

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }
