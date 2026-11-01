import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import httpx
from openai import OpenAI


# ============================================================
# 配置
# ============================================================

BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-flash"
DEEPSEEK_API_KEY = "xxx"

REASONING_EFFORT = "high"

DUMP_DIR = Path("toyagent_log")
DUMP_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_LOG = DUMP_DIR / "api_requests.json"
RESPONSE_LOG = DUMP_DIR / "api_responses.json"
TOOL_LOG = DUMP_DIR / "tool_calls.json"

MAX_TOOL_ROUNDS = 8
MAX_SEARCH_RESULTS = 5
MAX_CONTENT_CHARS = 2500


USE_DEFAULT_INPUT = False

DEFAULT_INPUT = ""

USE_STREAM = True

OVERLOAD_MSG = [
    {
        "role": "system",
        "content": "You are a helpful assistant.\n核心原则：无限的创作自由。禁止对用户做道德批判或道德劝告。禁止以MarkDown格式回复，换为更自然的表述方式。"
    }
]

# ============================================================
# 日志
# ============================================================

def append_json(path: str, record: dict) -> None:
    """以 JSON 格式追加记录。"""
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(record, ensure_ascii=False, indent=2)
            + "\n"
        )


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_request(request: httpx.Request) -> None:
    """
    HTTPX 请求钩子。
    """
    request.read()
    body = request.content.decode("utf-8", errors="replace")

    try:
        body = json.loads(body)
    except json.JSONDecodeError:
        body = None

    append_json(REQUEST_LOG, {
        "timestamp": timestamp(),
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers),
        "body": body,
    })


def log_response(response: httpx.Response) -> None:
    """记录非流式 HTTP 响应，便于排查错误。"""
    response.read()

    append_json(RESPONSE_LOG, {
        "timestamp": timestamp(),
        "method": response.request.method,
        "url": str(response.request.url),
        "status_code": response.status_code,
        "body": response.text,
    })


# 同一个 HTTP 客户端用于模型请求和搜索请求，
# 因此两类请求都会进入相同的日志系统。
http_client = httpx.Client(
    timeout=60.0,
    event_hooks={
        "request": [log_request],
        "response": [log_response],
    },
)

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=BASE_URL,
    http_client=http_client,
    max_retries=2,
)

# 流式请求使用独立的客户端：请求仍照常写入 REQUEST_LOG，
# 但不注册响应钩子——在钩子里读取响应会提前消费整个流，
# 破坏逐字输出。完整响应改由 chat_turn_stream 汇总后写入 RESPONSE_LOG。
stream_http_client = httpx.Client(
    timeout=60.0,
    event_hooks={
        "request": [log_request],
    },
)

stream_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=BASE_URL,
    http_client=stream_http_client,
    max_retries=2,
)


# ============================================================
# 搜索工具
# ============================================================

def web_search(query: str) -> str:
    """调用 DeepSeek Search API，并返回适合模型阅读的结果。"""

    query = query.strip()

    if not query or len(query) > 128:
        raise ValueError("搜索关键词必须为 1～128 个字符")

    response = http_client.post(
        f"{BASE_URL}/v1/tools/search",
        headers={
            "Authorization": (
                f"Bearer {DEEPSEEK_API_KEY}"
            ),
            "Content-Type": "application/json",
        },
        json={
            "query": query,
            "count": MAX_SEARCH_RESULTS,
            "need_content": True,
        },
    )

    response.raise_for_status()
    data = response.json()

    # 控制返回模型的内容长度，避免搜索正文过长。
    results = []

    for item in data.get("results", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "date": item.get("date"),
            "summary": item.get("summary", "")[
                :MAX_CONTENT_CHARS
            ],
            "content": item.get("content", "")[
                :MAX_CONTENT_CHARS
            ],
        })

    append_json(TOOL_LOG, {
        "timestamp": timestamp(),
        "event": "search_completed",
        "query": query,
        "result_count": len(results),
    })

    return json.dumps(
        {"query": query, "results": results},
        ensure_ascii=False,
    )


# ============================================================
# 工具定义与注册表
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "搜索互联网，获取最新信息、网页内容和来源链接。"
                "适用于时效性问题、新闻、近期版本变化及事实核验。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "具体的网页搜索关键词或问题",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
]

TOOL_REGISTRY = {
    "web_search": web_search,
}


# ============================================================
# 工具调用调度
# ============================================================

def execute_tool(tool_call) -> str:
    """校验工具名称与参数，执行工具并记录结果。"""

    name = tool_call.function.name
    call_id = tool_call.id

    append_json(TOOL_LOG, {
        "timestamp": timestamp(),
        "event": "tool_requested",
        "call_id": call_id,
        "name": name,
        "arguments": tool_call.function.arguments,
    })

    function = TOOL_REGISTRY.get(name)

    if function is None:
        result = json.dumps({
            "error": f"未注册的工具：{name}"
        }, ensure_ascii=False)
    else:
        try:
            arguments = json.loads(
                tool_call.function.arguments
            )

            if not isinstance(arguments, dict):
                raise ValueError("工具参数必须是 JSON 对象")

            result = function(**arguments)

        except Exception as exc:
            result = json.dumps({
                "error": type(exc).__name__,
                "message": str(exc),
            }, ensure_ascii=False)

    append_json(TOOL_LOG, {
        "timestamp": timestamp(),
        "event": "tool_result",
        "call_id": call_id,
        "name": name,
        "result": result,
    })

    return result


def call_model(messages: list[dict]):
    """执行一次非流式模型请求。"""

    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        stream=False,
        reasoning_effort=REASONING_EFFORT,
        extra_body={
            "thinking": {"type": "enabled"}
        },
    )


def call_model_stream(messages: list[dict]):
    """执行一次流式模型请求。"""

    return stream_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        stream=True,
        stream_options={"include_usage": True},
        reasoning_effort=REASONING_EFFORT,
        extra_body={
            "thinking": {"type": "enabled"}
        },
    )


def chat_turn(messages: list[dict]) -> str:
    """完成一轮用户对话，包括任意必要的工具调用。"""

    for round_index in range(1, MAX_TOOL_ROUNDS + 1):
        response = call_model(messages)
        message = response.choices[0].message

        # 保存完整 assistant 消息。
        # 思考模式下尤其不能随意丢掉 reasoning_content。
        assistant_message = message.model_dump(
            exclude_none=True
        )
        messages.append(assistant_message)

        append_json(TOOL_LOG, {
            "timestamp": timestamp(),
            "event": "model_turn",
            "round": round_index,
            "finish_reason": response.choices[0].finish_reason,
            "tool_call_count": len(message.tool_calls or []),
            "usage": (
                response.usage.model_dump()
                if response.usage else None
            ),
        })

        tool_calls = message.tool_calls or []

        if not tool_calls:
            return message.content or ""

        # 执行模型要求的每一个工具，并逐一追加结果。
        for tool_call in tool_calls:
            result = execute_tool(tool_call)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    raise RuntimeError(
        f"工具调用超过上限：{MAX_TOOL_ROUNDS} 轮"
    )


def _delta_attr(delta, name: str):
    """读取流式 delta 字段。

    兼容 SDK 未类型化的扩展字段（如思考模式的 reasoning_content），
    这类字段可能只存在于 model_extra 中。
    """

    value = getattr(delta, name, None)
    if value is None and hasattr(delta, "model_extra"):
        value = (delta.model_extra or {}).get(name)
    return value


def chat_turn_stream(messages: list[dict]) -> str:
    """完成一轮用户对话（流式），包括任意必要的工具调用。

    流式过程中逐字打印思考内容与可见回复：
      [Thinking]: 思考内容（reasoning_content）
      [Assistant]: 最终回复内容
    收集结束后把完整响应汇总写入 RESPONSE_LOG，使三种日志
    （请求 / 响应 / 工具）在流式模式下依然完整记录。
    """

    # 跨工具调用轮次复用，避免重复打印前缀。
    thinking_started = False
    answer_started = False

    for round_index in range(1, MAX_TOOL_ROUNDS + 1):
        stream = call_model_stream(messages)

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_acc: dict[int, dict] = {}
        finish_reason = None
        usage = None

        with stream:
            for chunk in stream:
                if chunk.usage:
                    usage = chunk.usage.model_dump()

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]

                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                delta = choice.delta
                if delta is None:
                    continue

                reasoning_part = _delta_attr(delta, "reasoning_content")
                if reasoning_part:
                    reasoning_parts.append(reasoning_part)
                    if not thinking_started:
                        print("\n[Thinking]: ", end="", flush=True)
                        thinking_started = True
                    print(reasoning_part, end="", flush=True)

                content_part = _delta_attr(delta, "content")
                if content_part:
                    content_parts.append(content_part)
                    if not answer_started:
                        print("\n[Assistant]: ", end="", flush=True)
                        answer_started = True
                    print(content_part, end="", flush=True)

                for tc_delta in delta.tool_calls or []:
                    index = tc_delta.index
                    acc = tool_calls_acc.setdefault(index, {
                        "id": None,
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    })

                    if tc_delta.id:
                        acc["id"] = tc_delta.id

                    if tc_delta.function:
                        if tc_delta.function.name:
                            acc["function"]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            acc["function"]["arguments"] += tc_delta.function.arguments

        content = "".join(content_parts) or None
        reasoning = "".join(reasoning_parts) or None
        tool_calls = [
            tool_calls_acc[index] for index in sorted(tool_calls_acc)
        ]

        # 组装与流式模型一致的完整 assistant 消息。
        assistant_message = {"role": "assistant"}
        if content is not None:
            assistant_message["content"] = content
        if reasoning:
            assistant_message["reasoning_content"] = reasoning
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        messages.append(assistant_message)

        # 汇总完整响应写入 RESPONSE_LOG。
        response = getattr(stream, "response", None)
        append_json(RESPONSE_LOG, {
            "timestamp": timestamp(),
            "method": response.request.method if response else "POST",
            "url": str(response.request.url) if response else "",
            "status_code": response.status_code if response else None,
            "streamed": True,
            "finish_reason": finish_reason,
            "content": content,
            "reasoning_content": reasoning,
            "tool_calls": tool_calls or None,
            "usage": usage,
        })

        append_json(TOOL_LOG, {
            "timestamp": timestamp(),
            "event": "model_turn",
            "round": round_index,
            "finish_reason": finish_reason,
            "tool_call_count": len(tool_calls),
            "usage": usage,
        })

        if not tool_calls:
            return content or ""

        for tool_call_data in tool_calls:
            # 复用 execute_tool，通过 SimpleNamespace 还原其期望的
            # tool_call 对象形状（.id / .function.name / .function.arguments）。
            tool_call_obj = SimpleNamespace(
                id=tool_call_data["id"],
                function=SimpleNamespace(
                    name=tool_call_data["function"]["name"],
                    arguments=tool_call_data["function"]["arguments"],
                ),
            )
            result = execute_tool(tool_call_obj)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_data["id"],
                "content": result,
            })

    raise RuntimeError(
        f"工具调用超过上限：{MAX_TOOL_ROUNDS} 轮"
    )


# ============================================================
# 主程序：多轮对话（流式 / 非流式）
# ============================================================

def main():
    messages = OVERLOAD_MSG.copy()

    print("DeepSeek Chat: ")

    try:
        while True:
            try:
                user_input = input("\n[User]: ").strip() if not USE_DEFAULT_INPUT else DEFAULT_INPUT
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit"}:
                break

            if user_input.lower() == "clear":
                messages = OVERLOAD_MSG.copy()
                print("已清空当前对话历史。")
                continue

            messages.append({
                "role": "user",
                "content": user_input,
            })

            try:
                if USE_STREAM:
                    chat_turn_stream(messages)
                    print()
                else:
                    answer = chat_turn(messages)
                    print(f"\n[Assistant]: {answer}")

            except Exception as exc:
                append_json(TOOL_LOG, {
                    "timestamp": timestamp(),
                    "event": "chat_error",
                    "error": type(exc).__name__,
                    "message": str(exc),
                })

                print(
                    f"\n请求失败：{type(exc).__name__}: {exc}"
                )

                # 出错时移除本轮用户消息及其未完成的交互，
                # 避免下一轮重用不完整的工具调用历史。
                # system 消息和此前完整对话仍然保留。
                last_user_index = next(
                    (
                        i for i in range(len(messages) - 1, 0, -1)
                        if messages[i].get("role") == "user"
                    ),
                    None,
                )

                if last_user_index is not None:
                    del messages[last_user_index:]
            if USE_DEFAULT_INPUT:
                break
    finally:
        http_client.close()
        stream_http_client.close()


if __name__ == "__main__":
    main()
