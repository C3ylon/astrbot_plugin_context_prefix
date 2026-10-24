import json
from functools import wraps
from typing import Any

from openai.resources.chat.completions import AsyncCompletions

from astrbot.api import star


@star.register(
    "astrbot_plugin_context_prefix",
    "AstrBot",
    "Replace the generated system message and prepend fixed messages to Chat Completions requests",
    "1.0.0",
)
class ContextPrefixPlugin(star.Star):
    """Replace the generated system message and prepend configured messages."""

    def __init__(self, context: star.Context, config: dict | None = None) -> None:
        """Initialize the request wrapper and apply the current plugin config.

        Args:
            context: AstrBot plugin context.
            config: Plugin configuration from the WebUI.
        """
        super().__init__(context, config)
        current_create = AsyncCompletions.create
        if getattr(current_create, "_astrbot_context_prefix", False):
            current_create._astrbot_context_prefix_config = config or {}
            return

        original_create = current_create

        @wraps(original_create)
        async def create_with_context_prefix(
            client: Any, *args: Any, **kwargs: Any
        ) -> Any:
            plugin_config = create_with_context_prefix._astrbot_context_prefix_config
            raw_messages = plugin_config.get("messages", "[]")
            try:
                prefix_messages = (
                    json.loads(raw_messages)
                    if isinstance(raw_messages, str)
                    else raw_messages
                )
            except json.JSONDecodeError:
                prefix_messages = []

            if not isinstance(prefix_messages, list) or any(
                not isinstance(message, dict)
                or not isinstance(message.get("role"), str)
                or "content" not in message
                for message in prefix_messages
            ):
                prefix_messages = []
            else:
                prefix_messages = [dict(message) for message in prefix_messages]

            messages = kwargs.get("messages")
            if prefix_messages and isinstance(messages, list):
                updated_kwargs = dict(kwargs)
                updated_messages = [
                    dict(message) if isinstance(message, dict) else message
                    for message in messages
                ]
                while (
                    updated_messages
                    and isinstance(updated_messages[0], dict)
                    and updated_messages[0].get("role") == "system"
                ):
                    updated_messages.pop(0)
                updated_messages[0:0] = prefix_messages
                updated_kwargs["messages"] = updated_messages
                kwargs = updated_kwargs

            return await original_create(client, *args, **kwargs)

        create_with_context_prefix._astrbot_context_prefix = True
        create_with_context_prefix._astrbot_context_prefix_config = config or {}
        AsyncCompletions.create = create_with_context_prefix  # type: ignore[method-assign]
