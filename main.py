import asyncio
import json
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

from openai.resources.chat.completions import AsyncCompletions

from astrbot.api import star


@star.register(
    "astrbot_plugin_context_prefix",
    "AstrBot",
    "Customize and dump outbound Chat Completions requests",
    "1.0.0",
)
class ContextPrefixPlugin(star.Star):
    """Customize messages and headers, and record outbound requests."""

    def __init__(self, context: star.Context, config: dict | None = None) -> None:
        """Initialize request customization and load local configuration.

        Args:
            context: AstrBot plugin context.
            config: Legacy plugin configuration, retained for loader compatibility.
        """
        super().__init__(context, config)
        plugin_dir = Path(__file__).resolve().parent
        self.dump_enabled = True
        self.dump_path = plugin_dir / "requests.json"
        self.dump_lock = asyncio.Lock()
        self.dump_hook_unavailable_warned = False

        dump_config_path = plugin_dir / "dump.json"
        try:
            dump_config = json.loads(
                dump_config_path.read_text(encoding="utf-8-sig")
            )
        except (OSError, json.JSONDecodeError) as exc:
            self.logger.error(
                "Failed to load dump configuration from %s: %s",
                dump_config_path,
                exc,
            )
            dump_config = {}

        if not isinstance(dump_config, dict):
            self.logger.error(
                "Invalid dump configuration in %s; expected a JSON object.",
                dump_config_path,
            )
            dump_config = {}

        dump_enabled = dump_config.get("enabled", True)
        if isinstance(dump_enabled, bool):
            self.dump_enabled = dump_enabled
        else:
            self.logger.warning(
                "Ignoring non-boolean dump enabled value in %s.", dump_config_path
            )

        configured_dump_path = dump_config.get("path", "requests.json")
        if isinstance(configured_dump_path, str) and configured_dump_path.strip():
            configured_dump_path = Path(configured_dump_path).expanduser()
            self.dump_path = (
                configured_dump_path
                if configured_dump_path.is_absolute()
                else plugin_dir / configured_dump_path
            ).resolve()
        else:
            self.logger.warning(
                "Ignoring invalid dump path in %s; using %s.",
                dump_config_path,
                self.dump_path,
            )

        if self.dump_enabled:
            self.logger.info("Request dumping enabled: %s", self.dump_path)
        else:
            self.logger.info("Request dumping disabled by configuration.")

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
                "Invalid message prefix in %s; expected a JSON array of role/content "
                "objects.",
                messages_path,
            )
            prefix_messages = []
        else:
            prefix_messages = [dict(message) for message in prefix_messages]

        headers_path = Path(__file__).with_name("headers.json")
        try:
            headers_config = json.loads(headers_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            self.logger.error(
                "Failed to load header overrides from %s: %s", headers_path, exc
            )
            headers_config = {}

        if not isinstance(headers_config, dict):
            self.logger.error(
                "Invalid header configuration in %s; expected an object with a "
                "headers object.",
                headers_path,
            )
            header_overrides = {}
        else:
            header_overrides = headers_config.get("headers", {})
            if not isinstance(header_overrides, dict):
                self.logger.error(
                    "Invalid header configuration in %s; expected an object with "
                    "a headers object.",
                    headers_path,
                )
                header_overrides = {}

            valid_header_overrides = {}
            for name, value in header_overrides.items():
                if not isinstance(name, str) or not name.strip():
                    self.logger.warning(
                        "Ignoring an invalid header name in %s.", headers_path
                    )
                    continue
                if value is not None and not isinstance(value, str):
                    self.logger.warning(
                        "Ignoring non-string header value for %s in %s.",
                        name,
                        headers_path,
                    )
                    continue
                valid_header_overrides[name] = value
            header_overrides = valid_header_overrides

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
            sdk_client = getattr(client, "_client", None)
            http_client = getattr(sdk_client, "_client", None)
            if not self._attach_request_hooks(
                http_client, header_overrides, self.dump_enabled
            ):
                if self.dump_enabled and not self.dump_hook_unavailable_warned:
                    self.logger.warning(
                        "Could not attach request hooks to the OpenAI HTTP client; "
                        "request dumping is unavailable."
                    )
                    self.dump_hook_unavailable_warned = True
                if header_overrides:
                    updated_kwargs = dict(kwargs)
                    extra_headers = dict(updated_kwargs.get("extra_headers") or {})
                    for name, value in header_overrides.items():
                        for existing_name in list(extra_headers):
                            if (
                                isinstance(existing_name, str)
                                and existing_name.lower() == name.lower()
                            ):
                                del extra_headers[existing_name]
                        if value is not None:
                            extra_headers[name] = value
                    updated_kwargs["extra_headers"] = extra_headers
                    kwargs = updated_kwargs

            messages = kwargs.get("messages")
            if prefix_messages and isinstance(messages, list):
                updated_kwargs = dict(kwargs)
                updated_messages = [
                    dict(message) if isinstance(message, dict) else message
                    for message in messages
                ]

                # Remove complete prefix copies already present at the head. This
                # keeps injection idempotent if a caller reuses sent messages.
                while True:
                    leading_system_count = 0
                    while (
                        leading_system_count < len(updated_messages)
                        and isinstance(
                            updated_messages[leading_system_count], dict
                        )
                        and updated_messages[leading_system_count].get("role")
                        == "system"
                    ):
                        leading_system_count += 1

                    existing_prefix_start = None
                    for start in range(leading_system_count + 1):
                        end = start + len(prefix_messages)
                        if end > len(updated_messages):
                            continue
                        if all(
                            isinstance(existing, dict)
                            and all(
                                existing.get(key) == value
                                for key, value in prefix_message.items()
                            )
                            for existing, prefix_message in zip(
                                updated_messages[start:end], prefix_messages
                            )
                        ):
                            existing_prefix_start = start
                            break

                    if existing_prefix_start is None:
                        del updated_messages[:leading_system_count]
                        break

                    del updated_messages[
                        existing_prefix_start : existing_prefix_start
                        + len(prefix_messages)
                    ]

                updated_messages[0:0] = [
                    dict(message) for message in prefix_messages
                ]
                updated_kwargs["messages"] = updated_messages
                kwargs = updated_kwargs

            return await original_create(client, *args, **kwargs)

        create_with_context_prefix._astrbot_context_prefix = True
        create_with_context_prefix._astrbot_context_prefix_original = original_create
        setattr(AsyncCompletions, "create", create_with_context_prefix)

    def _attach_request_hooks(
        self,
        http_client: Any,
        header_overrides: dict[str, str | None],
        dump_enabled: bool,
    ) -> bool:
        """Attach header and request-dump hooks to an HTTPX client.

        Args:
            http_client: HTTPX client used by the OpenAI SDK.
            header_overrides: Header values to set, or None to remove them.
            dump_enabled: Whether matching requests should be written to JSON.

        Returns:
            Whether the client exposes a compatible HTTPX request-hook interface.
        """
        event_hooks = getattr(http_client, "event_hooks", None)
        if not isinstance(event_hooks, dict):
            return False

        header_hook_attribute = "_astrbot_context_prefix_header_hook"
        header_overrides_attribute = "_astrbot_context_prefix_header_overrides"
        header_hook = getattr(http_client, header_hook_attribute, None)
        if header_hook is None and header_overrides:

            async def apply_header_overrides(request: Any) -> None:
                """Apply configured changes to the final HTTP headers.

                Args:
                    request: Outbound HTTP request to modify.
                """
                if (
                    request.method.upper() != "POST"
                    or not request.url.path.rstrip("/").endswith(
                        "/chat/completions"
                    )
                ):
                    return

                overrides = getattr(
                    http_client,
                    header_overrides_attribute,
                    {},
                )
                for name, value in overrides.items():
                    if value is None:
                        if name in request.headers:
                            del request.headers[name]
                    else:
                        request.headers[name] = value

            event_hooks.setdefault("request", []).insert(0, apply_header_overrides)
            setattr(http_client, header_hook_attribute, apply_header_overrides)
            header_hook = apply_header_overrides

        if header_hook is not None:
            setattr(
                http_client,
                header_overrides_attribute,
                dict(header_overrides),
            )

        dump_hook_attribute = "_astrbot_context_prefix_dump_hook"
        dump_state_attribute = "_astrbot_context_prefix_dump_state"
        dump_hook = getattr(http_client, dump_hook_attribute, None)
        if dump_hook is None and dump_enabled:

            async def capture_request(request: Any) -> None:
                """Append one serialized Chat Completions request to JSON.

                Args:
                    request: Outbound HTTP request to capture.
                """
                if (
                    request.method.upper() != "POST"
                    or not request.url.path.rstrip("/").endswith(
                        "/chat/completions"
                    )
                ):
                    return

                state = getattr(http_client, dump_state_attribute, {})
                if not state.get("enabled", True):
                    return
                try:
                    raw_body = await request.aread()
                    try:
                        body = json.loads(raw_body)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        body = raw_body.decode("utf-8", errors="replace")

                    headers = {
                        name: (
                            "<redacted>"
                            if name.lower()
                            in {
                                "authorization",
                                "api-key",
                                "x-api-key",
                                "cookie",
                            }
                            else value
                        )
                        for name, value in request.headers.items()
                    }
                    record = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "method": request.method,
                        "url": str(request.url),
                        "headers": headers,
                        "body": body,
                    }
                    async with state["lock"]:
                        state["path"].parent.mkdir(
                            parents=True,
                            exist_ok=True,
                        )
                        with state["path"].open("a", encoding="utf-8") as dump_file:
                            dump_file.write(
                                json.dumps(record, ensure_ascii=False) + "\n"
                            )
                except Exception:
                    logger = state.get("logger")
                    if logger:
                        logger.exception("Failed to dump an outbound API request.")

            event_hooks.setdefault("request", []).append(capture_request)
            setattr(http_client, dump_hook_attribute, capture_request)
            self.logger.info("Attached outbound request dumper to HTTP client.")

        setattr(
            http_client,
            dump_state_attribute,
            {
                "enabled": dump_enabled,
                "path": self.dump_path,
                "lock": self.dump_lock,
                "logger": self.logger,
            },
        )
        return True
