import json
from datetime import datetime, timezone
from pathlib import Path
import httpx
from openai import OpenAI


# ============================================================
# 配置
# ============================================================

BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-flash"
DEEPSEEK_API_KEY = "xxx"

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
        reasoning_effort="high",
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


# ============================================================
# 主程序：非流式多轮对话
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


if __name__ == "__main__":
    main()
