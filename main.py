import json
from functools import wraps
from pathlib import Path
from typing import Any

from openai.resources.chat.completions import AsyncCompletions

from astrbot.api import star


@star.register(
    "astrbot_plugin_context_prefix",
    "AstrBot",
    "Replace the generated system message and prepend messages from messages.json",
    "1.0.0",
)
class ContextPrefixPlugin(star.Star):
    """Replace the generated system message and prepend messages from a file."""

    def __init__(self, context: star.Context, config: dict | None = None) -> None:
        """Initialize the request wrapper and load the configured message prefix.

        Args:
            context: AstrBot plugin context.
            config: Legacy plugin configuration, retained for loader compatibility.
        """
        super().__init__(context, config)
        messages_path = Path(__file__).with_name("messages.json")
        try:
            prefix_messages = json.loads(messages_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            self.logger.error(
                "Failed to load message prefix from %s: %s", messages_path, exc
            )
            prefix_messages = []

        if not isinstance(prefix_messages, list) or any(
            not isinstance(message, dict)
            or not isinstance(message.get("role"), str)
            or "content" not in message
            for message in prefix_messages
        ):
            self.logger.error(
                "Invalid message prefix in %s; expected a JSON array of role/content objects.",
                messages_path,
            )
            prefix_messages = []
        else:
            prefix_messages = [dict(message) for message in prefix_messages]

        current_create = AsyncCompletions.create
        original_create = getattr(
            current_create,
            "_astrbot_context_prefix_original",
            current_create,
        )

        @wraps(original_create)
        async def create_with_context_prefix(
            client: Any, *args: Any, **kwargs: Any
        ) -> Any:
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
        create_with_context_prefix._astrbot_context_prefix_original = original_create
        AsyncCompletions.create = create_with_context_prefix  # type: ignore[method-assign]
