#!/usr/bin/env python3
"""PLANARIA_ONE single-file backend engine (v7 - Agent-friendly refactor).

This file is intentionally self-contained so that deployment only needs:
- planaria_setup.sh
- planaria_run.sh
- planaria_core.py

Architecture (class order in this file):
  1. Constants & Utilities      – safe_str, TOOL_SPECS, helper functions
  2. WorkspaceManager           – workspace init, config loading
  3. MemoryStore                – GraphRAG-style memory
  4. WorkspaceTools             – tool implementations (file, shell, web, skill, memory)
  5. LLMClient                  – HTTP client with route/cooldown/fallback
  6. IntentClassifier            – all keyword-based intent detection (single source of truth)
  7. ToolCallParser             – parse/normalize tool calls from LLM output
  8. ContextManager             – context window compression & sanitization
  9. PromptBuilder              – system/task prompt assembly
 10. ToolDispatcher             – data-driven tool dispatch (registry, not if/elif)
 11. JobBreakerEngine           – planning & task decomposition
 12. TaskExecutor               – tool-calling execution loop
 13. AgentEngine                – thin orchestrator wiring all components
 14. SchedulerRunner            – background task scheduler
 15. TelegramRunner             – Telegram bot interface
 16. CLIRunner                  – CLI interface
 17. main()                     – entry point
"""

from __future__ import annotations

import argparse
import email as email_mod
import imaplib
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.header import decode_header as _decode_header_raw
from pathlib import Path
from typing import Any, Callable

import requests

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Constants & Utilities
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_MODEL = "openai/gpt-4o"
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
KETI_CONV_API_BASE = os.environ.get("PLANARIA_KETI_CONV_API_BASE", "")
KETI_CONV_MODEL = os.environ.get("PLANARIA_KETI_CONV_MODEL", "nota-ai/Solar-Open-100B-NotaMoEQuant-Int4")
MAX_TOOL_ITERS = 40
REQUEST_TIMEOUT = 90
DEFAULT_CONTEXT_MAX_CHARS = 20000
DEFAULT_COMPRESSED_CONTEXT_CHARS = 12000
LLM_QUERY_TIMEOUT_SEC = 20
LLM_QUERY_TIMEOUT_MAX_SEC = 40
LLM_QUERY_TIMEOUT_STEP_SEC = 10
DEFAULT_ROUTE_COOLDOWN_SEC = 6 * 60 * 60


def _decode_email_header(header_val: str | None) -> str:
    """Decode RFC 2047 encoded email header value."""
    if not header_val:
        return ""
    parts: list[str] = []
    for part, charset in _decode_header_raw(header_val):
        if isinstance(part, bytes):
            parts.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            parts.append(part)
    return " ".join(parts)


def _extract_email_body(msg: email_mod.message.Message) -> str:
    """Extract plain text body from an email message."""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    return payload.decode(part.get_content_charset() or "utf-8", errors="replace")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            return payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
    return ""


def safe_str(s: Any) -> str:
    """Safely convert to string and handle surrogates that cause JSON encoding errors."""
    if not isinstance(s, str):
        s = str(s)
    return s.encode("utf-8", errors="replace").decode("utf-8")


def safe_json_dumps(obj: Any, **kwargs: Any) -> str:
    """json.dumps that never fails on surrogate characters.

    The SRT skill (and other external tools) can produce stdout with broken
    encoding (e.g. \\udceb). Standard json.dumps with ensure_ascii=False will
    raise UnicodeEncodeError on these. This wrapper catches that and falls back
    to ensure_ascii=True which escapes non-ASCII characters instead.
    """
    kwargs.setdefault("ensure_ascii", False)
    try:
        return json.dumps(obj, **kwargs)
    except UnicodeEncodeError:
        # Strip surrogates from all string values and retry
        def _clean(v: Any) -> Any:
            if isinstance(v, str):
                return v.encode("utf-8", errors="replace").decode("utf-8")
            if isinstance(v, dict):
                return {_clean(k): _clean(val) for k, val in v.items()}
            if isinstance(v, list):
                return [_clean(i) for i in v]
            return v
        return json.dumps(_clean(obj), **kwargs)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_json_load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# SecretStore: OS keyring with secrets.json fallback
# ═══════════════════════════════════════════════════════════════════════════════

_SECRET_REF_PREFIX = "@secret:"


class SecretStore:
    """Credential storage. Tries OS keyring first, falls back to secrets.json (chmod 600)."""

    def __init__(self, workspace: Path, agent_name: str):
        self.workspace = workspace
        self.service = f"planaria_{agent_name}"
        self.secrets_path = workspace / "secrets.json"
        self._use_keyring = self._probe_keyring()

    @staticmethod
    def _probe_keyring() -> bool:
        try:
            import keyring
            from keyring.backends import fail as _fail_backend
            backend = keyring.get_keyring()
            backend_cls = type(backend).__module__ + "." + type(backend).__name__
            # Only accept real OS backends (SecretService, macOS Keychain, Windows)
            # Reject file-based, encrypted, plaintext, or fail backends
            if any(x in backend_cls.lower() for x in ("alt", "encrypted", "plaintext", "fail", "null", "chainer")):
                return False
            if isinstance(backend, _fail_backend.Keyring):
                return False
            return True
        except Exception:
            return False

    def store(self, key: str, value: str) -> None:
        """Store a secret value."""
        if self._use_keyring:
            import keyring
            keyring.set_password(self.service, key, value)
        else:
            data = safe_json_load(self.secrets_path, {})
            data[key] = value
            write_json(self.secrets_path, data)
            try:
                import os
                os.chmod(str(self.secrets_path), 0o600)
            except OSError:
                pass

    def load(self, key: str) -> str | None:
        """Load a secret value. Returns None if not found."""
        if self._use_keyring:
            try:
                import keyring
                return keyring.get_password(self.service, key)
            except Exception:
                return None
        data = safe_json_load(self.secrets_path, {})
        return data.get(key)

    def delete(self, key: str) -> bool:
        """Delete a secret. Returns True if deleted."""
        if self._use_keyring:
            try:
                import keyring
                keyring.delete_password(self.service, key)
                return True
            except Exception:
                return False
        data = safe_json_load(self.secrets_path, {})
        if key in data:
            del data[key]
            write_json(self.secrets_path, data)
            return True
        return False

    @property
    def backend_name(self) -> str:
        return "OS Keyring" if self._use_keyring else "secrets.json (파일 보호)"

    @staticmethod
    def make_ref(key: str) -> str:
        """Create a config reference string for a secret key."""
        return f"{_SECRET_REF_PREFIX}{key}"

    @staticmethod
    def is_ref(value: str) -> bool:
        """Check if a config value is a secret reference."""
        return isinstance(value, str) and value.startswith(_SECRET_REF_PREFIX)

    @staticmethod
    def ref_key(value: str) -> str:
        """Extract the key name from a secret reference."""
        return value[len(_SECRET_REF_PREFIX):]


def resolve_secrets(data: Any, secret_store: SecretStore) -> Any:
    """Recursively resolve @secret: references in config data."""
    if isinstance(data, str) and SecretStore.is_ref(data):
        return secret_store.load(SecretStore.ref_key(data)) or ""
    if isinstance(data, dict):
        return {k: resolve_secrets(v, secret_store) for k, v in data.items()}
    if isinstance(data, list):
        return [resolve_secrets(v, secret_store) for v in data]
    return data


SECURITY_NOTICE = (
    "⚠️ 보안 안내\n"
    "입력하신 API 키와 비밀번호는 {backend}에 안전하게 저장됩니다.\n"
    "config.json에는 실제 값 대신 참조(@secret:...)만 남습니다.\n\n"
    "중요: OS 계정(로그인 비밀번호)이 보안의 핵심입니다.\n"
    "  • OS 비밀번호를 타인과 공유하지 마세요\n"
    "  • 서버라면 SSH 키 인증을 사용하세요\n"
    "  • 공유 PC에서는 로그아웃을 철저히 하세요\n"
)


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "update_config",
            "description": "Update the agent's config.json file (e.g., to save user-provided API keys like 'brave_api_key').",
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {"type": "string", "description": "Config section (e.g., 'search', 'llm', 'telegram')"},
                    "key": {"type": "string", "description": "Config key (e.g., 'brave_api_key', 'api_key')"},
                    "value": {"type": "string", "description": "The value to set"}
                },
                "required": ["section", "key", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_hub",
            "description": "Search ClawHub for installable external skills by keywords. Returns repos with clone_url and install_cmd.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords to search (e.g., 'weather skill', 'srt booking')"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using Brave Search. Use search_type='news' for breaking news, current events, stock/market updates. Use search_type='knowledge' for technical docs, how-to, research, general facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "search_type": {"type": "string", "enum": ["news", "knowledge", "auto"], "description": "Search mode: 'news' for current events/뉴스, 'knowledge' for technical/research, 'auto' to detect automatically. Default: auto."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_task",
            "description": "Schedule a task at a UTC time. Supports one-shot and recurring schedules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "run_at_utc": {"type": "string", "description": "ISO 8601 UTC timestamp (e.g., '2026-03-17T21:00:00Z')"},
                    "task_prompt": {"type": "string", "description": "The prompt to execute at that time."},
                    "recurrence": {
                        "type": "string",
                        "enum": ["auto", "none", "daily", "weekly", "monthly"],
                        "description": "Recurrence rule. 'auto' infers from task_prompt keywords (매일/매주/매월)."
                    },
                    "interval": {"type": "integer", "description": "Recurrence interval (default 1)."},
                },
                "required": ["run_at_utc", "task_prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_scheduled_tasks",
            "description": "List all scheduled tasks (pending and active) for this workspace. Use this when the user asks 'what tasks are scheduled', '예약된 작업이 뭐가 있어', '내 스케줄 보여줘', etc. NEVER fabricate task lists — always call this tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_done": {"type": "boolean", "description": "Include already-executed one-shot tasks. Default false."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_scheduled_task",
            "description": "Cancel a scheduled task by its task_id (returned from list_scheduled_tasks).",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "The task_id from list_scheduled_tasks."},
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files under a workspace-relative directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read text content from a workspace-relative file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write UTF-8 text content to a workspace-relative file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Search long-term memory entries for relevant items.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command inside workspace root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_sec": {"type": "integer"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "install_skill",
            "description": "Install an external skill from ClawHub: clone → validate (run.py/manifest) → install deps → register as tool. Use after search_hub.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_url": {"type": "string"},
                    "skill_name": {"type": "string"},
                },
                "required": ["repo_url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_skill",
            "description": "Run an installed skill script (run.sh or run.py).",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string"},
                    "args": {"type": "string"},
                },
                "required": ["skill_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": (
                "Store a durable memory fact into the knowledge graph. "
                "Layer rules (STRICT): "
                "L1 = Agent identity ONLY (name, personality, core behavior). NEVER store project/domain/user info in L1. "
                "L2 = User profile & preferences (ongoing projects, goals, preferences, personal info). "
                "L3 = Episodic facts (conversation-specific knowledge, transient info, task details). "
                "entity = the subject of the fact (e.g. 'user', 'agent', 'project_name'). "
                "relation = relationship type (e.g. 'identity', 'preference', 'goal', 'has_project', 'is_type'). "
                "target_entity = the object entity if this fact links two entities (e.g. 'python', 'stock_prediction'). "
                "IMPORTANT: Before storing, consider if the fact is truly worth remembering long-term. "
                "Do NOT store transient queries or search requests as facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "The fact to remember. Should be a clear, self-contained statement."},
                    "layer": {"type": "string", "enum": ["L1", "L2", "L3"], "description": "L1=agent identity ONLY, L2=user profile/prefs, L3=episodic"},
                    "relation": {"type": "string", "description": "Relationship type: identity, preference, goal, has_project, is_type, related_to, etc."},
                    "entity": {"type": "string", "description": "Subject entity of this fact"},
                    "target_entity": {"type": "string", "description": "Object entity if linking two entities, otherwise empty"},
                },
                "required": ["fact", "layer", "entity", "relation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_email",
            "description": "Check email inbox (Gmail). Returns recent emails with subject, sender, date, and snippet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "provider": {"type": "string", "enum": ["gmail"], "description": "Email provider to check"},
                    "limit": {"type": "integer", "description": "Max emails to return (default 5)"},
                    "unread_only": {"type": "boolean", "description": "Only return unread emails (default true)"},
                },
                "required": ["provider"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_calendar",
            "description": "Check Google Calendar for upcoming events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days_ahead": {"type": "integer", "description": "Number of days to look ahead (default 1, max 7)"},
                    "max_results": {"type": "integer", "description": "Max events to return (default 10)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reset_workspace",
            "description": "Reset workspace memory/logs/data/skills. Requires confirm='RESET'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "confirm": {"type": "string"},
                },
                "required": ["confirm"],
            },
        },
    },
]

INTERNAL_TOOL_NAMES = {
    str(item.get("function", {}).get("name", ""))
    for item in TOOL_SPECS
    if str(item.get("function", {}).get("name", ""))
}


def is_internal_tool_name(name: str) -> bool:
    return name in INTERNAL_TOOL_NAMES


def split_tool_calls_by_origin(tool_calls: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    internal_calls: list[dict[str, Any]] = []
    external_calls: list[dict[str, Any]] = []
    for call in tool_calls:
        fn_name = str(call.get("function", {}).get("name") or "")
        if is_internal_tool_name(fn_name):
            internal_calls.append(call)
        else:
            external_calls.append(call)
    return internal_calls, external_calls


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: WorkspaceManager
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RuntimeConfig:
    agent_name: str
    aliases: list[str]
    workspace: Path
    models: list[str]
    api_base: str
    api_key: str
    fallback_routes: list[dict[str, str]]
    route_cooldown_sec: int
    max_tool_iterations: int
    context_max_chars: int
    compressed_context_chars: int
    max_recent_messages: int
    request_timeout_sec: int
    telegram_enabled: bool
    telegram_token: str
    telegram_allow_from: list[str]
    brave_api_key: str
    integrations: dict[str, Any] = field(default_factory=dict)


class WorkspaceManager:
    def __init__(self, base_dir: Path, agent_name: str):
        self.base_dir = base_dir.resolve()
        self.agent_name = agent_name
        self.root = (self.base_dir / f"workspace_{agent_name}").resolve()
        self.memory_dir = self.root / "memory"
        self.skills_dir = self.root / "skills"
        self.logs_dir = self.root / "logs"
        self.data_dir = self.root / "data"
        self.config_path = self.root / "config.json"

    def ensure(self) -> None:
        is_new = not self.root.exists()
        self.root.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Copy default skills from workspace_default if this is a fresh workspace
        if is_new:
            default_skills = self.root.parent / "workspace_default" / "skills"
            if default_skills.is_dir():
                for skill_dir in default_skills.iterdir():
                    if skill_dir.is_dir():
                        dest = self.skills_dir / skill_dir.name
                        if not dest.exists():
                            shutil.copytree(str(skill_dir), str(dest))
        if not self.config_path.exists():
            template = {
                "agent": {"name": self.agent_name, "aliases": []},
                "llm": {
                    "models": [DEFAULT_MODEL],
                    "api_base": DEFAULT_API_BASE,
                    "api_key": "",
                    "fallback_routes": [],
                    "route_cooldown_sec": DEFAULT_ROUTE_COOLDOWN_SEC,
                    "max_tool_iterations": MAX_TOOL_ITERS,
                    "context_max_chars": DEFAULT_CONTEXT_MAX_CHARS,
                    "compressed_context_chars": DEFAULT_COMPRESSED_CONTEXT_CHARS,
                    "max_recent_messages": 16,
                    "request_timeout_sec": 180,
                },
                "telegram": {
                    "enabled": True,
                    "token": "",
                    "allow_from": ["*"],
                },
                "search": {
                    "brave_api_key": ""
                },
                "integrations": {}
            }
            write_json(self.config_path, template)

    def load_config(self) -> RuntimeConfig:
        self.ensure()
        cfg = safe_json_load(self.config_path, {})
        agent_cfg = cfg.get("agent", {})
        agent_name = str(agent_cfg.get("name") or self.agent_name)
        raw_aliases = agent_cfg.get("aliases", [])
        aliases: list[str] = []
        if isinstance(raw_aliases, list):
            aliases = [str(a).strip() for a in raw_aliases if str(a).strip()]
        display_name = str(agent_cfg.get("display_name") or "").strip()
        if display_name and display_name not in aliases:
            aliases.insert(0, display_name)

        llm = cfg.get("llm", {})
        tg = cfg.get("telegram", {})
        search_cfg = cfg.get("search", {})
        raw_models = llm.get("models")
        if not raw_models:
            legacy_model = llm.get("model")
            raw_models = [legacy_model] if legacy_model else [DEFAULT_MODEL]
        elif isinstance(raw_models, str):
            raw_models = [raw_models]
        models = [str(m) for m in raw_models if m]

        api_base = str(llm.get("api_base") or DEFAULT_API_BASE).rstrip("/")
        api_key = str(llm.get("api_key") or "")
        fallback_routes_raw = llm.get("fallback_routes") or []
        fallback_routes: list[dict[str, str]] = []
        if isinstance(fallback_routes_raw, list):
            for item in fallback_routes_raw:
                if not isinstance(item, dict):
                    continue
                model = str(item.get("model") or "").strip()
                route_base = str(item.get("api_base") or "").strip().rstrip("/")
                route_key = str(item.get("api_key") or "").strip()
                if not model or not route_base or not route_key:
                    continue
                fallback_routes.append(
                    {
                        "model": model,
                        "api_base": route_base,
                        "api_key": route_key,
                    }
                )
        route_cooldown_sec = int(llm.get("route_cooldown_sec") or DEFAULT_ROUTE_COOLDOWN_SEC)
        max_iters = int(llm.get("max_tool_iterations") or MAX_TOOL_ITERS)
        context_max_chars = int(llm.get("context_max_chars") or DEFAULT_CONTEXT_MAX_CHARS)
        compressed_context_chars = int(llm.get("compressed_context_chars") or DEFAULT_COMPRESSED_CONTEXT_CHARS)
        max_recent_messages = int(llm.get("max_recent_messages") or 16)
        request_timeout_sec = int(llm.get("request_timeout_sec") or 180)
        return RuntimeConfig(
            agent_name=agent_name,
            aliases=aliases,
            workspace=self.root,
            models=models,
            api_base=api_base,
            api_key=api_key,
            fallback_routes=fallback_routes,
            route_cooldown_sec=max(60, min(7 * 24 * 60 * 60, route_cooldown_sec)),
            max_tool_iterations=max(1, min(120, max_iters)),
            context_max_chars=max(4000, context_max_chars),
            compressed_context_chars=max(2000, min(compressed_context_chars, context_max_chars)),
            max_recent_messages=max(4, min(60, max_recent_messages)),
            request_timeout_sec=max(30, min(600, request_timeout_sec)),
            telegram_enabled=bool(tg.get("enabled", True)),
            telegram_token=str(tg.get("token") or ""),
            telegram_allow_from=[str(x) for x in tg.get("allow_from", ["*"])],
            brave_api_key=str(
                search_cfg.get("brave_api_key")
                or cfg.get("migrated_keys", {}).get("tools_web", {}).get("search", {}).get("apiKey")
                or ""
            ),
            integrations=cfg.get("integrations", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: MemoryStore
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryStore:
    """GraphRAG-style memory with strict layer hierarchy and relevance-gated retrieval.

    Layer semantics (STRICT):
      L1 - Agent identity ONLY (name, personality, core behavior rules).
           Stored once at bootstrap. Domain/project/user info NEVER goes here.
      L2 - User profile & preferences (ongoing projects, goals, personal info).
           Persistent across sessions. Updated, not duplicated.
      L3 - Episodic facts (conversation-specific knowledge, transient task details).
           Decays fastest; may be pruned.
    """

    _L1_RELATIONS = frozenset({"identity", "personality", "behavior_rule"})
    _LAYER_HUBS = {"L1": "__hub_l1__", "L2": "__hub_l2__", "L3": "__hub_l3__"}

    def __init__(self, workspace: Path):
        self.path = workspace / "memory" / "memory.jsonl"
        self.graph_path = workspace / "memory" / "graph_memory.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, role: str, content: str, meta: dict[str, Any] | None = None) -> None:
        line = {
            "ts": utc_now(),
            "role": role,
            "content": safe_str(content),
            "meta": meta or {},
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    def has_memory_fact(self, fact_substring: str, layer: str | None = None) -> bool:
        if not self.path.exists():
            return False
        needle = fact_substring.strip().lower()
        if not needle:
            return False
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(raw)
            except Exception:
                continue
            if str(row.get("role", "")) != "memory":
                continue
            meta = row.get("meta", {}) if isinstance(row.get("meta"), dict) else {}
            if layer and str(meta.get("layer", "")).upper() != layer.upper():
                continue
            content = str(row.get("content", "")).lower()
            if needle in content:
                return True
        return False

    def recent(self, limit: int = 12, source: str | None = None, user_id: str | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        rows = []
        for raw in lines:
            try:
                row = json.loads(raw)
            except Exception:
                continue
            meta = row.get("meta", {}) if isinstance(row.get("meta"), dict) else {}
            if source and str(meta.get("source") or "") != source:
                continue
            if user_id and str(meta.get("user_id") or "") != user_id:
                continue
            rows.append(row)
        if limit > 0:
            rows = rows[-limit:]
        return rows

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        text_lower = text.lower()
        raw_tokens = [t for t in text_lower.split() if len(t) >= 2]
        _KO_SUFFIXES = (
            "은", "는", "이", "가", "을", "를", "에", "의", "로", "와", "과",
            "도", "만", "에서", "에게", "으로", "이다", "입니다", "이고", "인",
            "한", "하는", "했다", "합니다", "이며", "으며",
        )
        normalized: set[str] = set()
        for tok in raw_tokens:
            normalized.add(tok)
            for suf in sorted(_KO_SUFFIXES, key=len, reverse=True):
                if tok.endswith(suf) and len(tok) > len(suf) + 1:
                    stem = tok[: -len(suf)]
                    if len(stem) >= 2:
                        normalized.add(stem)
                    break
        for tok in raw_tokens:
            for i in range(len(tok) - 1):
                bigram = tok[i : i + 2]
                normalized.add(bigram)
        return normalized

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        query_tokens = self._tokenize(query)
        if not query_tokens or not self.path.exists():
            return []
        scored: list[tuple[float, dict[str, Any]]] = []
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(raw)
            except Exception:
                continue
            text = str(row.get("content", ""))
            text_tokens = self._tokenize(text)
            sim = self._jaccard(query_tokens, text_tokens)
            if sim > 0.0:
                scored.append((sim, row))
        scored.sort(key=lambda x: (x[0], x[1].get("ts", "")), reverse=True)
        return [x[1] for x in scored[:limit]]

    def _is_duplicate_graph_fact(self, entity: str, fact: str, threshold: float = 0.75) -> bool:
        if not self.graph_path.exists():
            return False
        fact_tokens = self._tokenize(fact)
        entity_lower = entity.lower().strip()
        for raw in self.graph_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(raw)
                triple = row.get("triple", {})
                if triple.get("entity", "") != entity_lower and triple.get("target_entity", "") != entity_lower:
                    continue
                existing_tokens = self._tokenize(triple.get("fact", ""))
                if self._jaccard(fact_tokens, existing_tokens) >= threshold:
                    return True
            except Exception:
                continue
        return False

    def remember_fact(self, fact: str, layer: str = "L3", relation: str = "related_to", entity: str = "user", target_entity: str = "") -> dict[str, Any]:
        layer_norm = layer.upper().strip()
        if layer_norm not in {"L1", "L2", "L3"}:
            layer_norm = "L3"
        if layer_norm == "L1" and relation.lower().strip() not in self._L1_RELATIONS:
            layer_norm = "L2"
        entity_norm = entity.lower().strip() or "user"
        relation_norm = relation.lower().strip() or "related_to"
        target_norm = target_entity.lower().strip() if target_entity else ""
        # Enforce graph connectivity by ensuring every triple has a target node.
        # If missing, connect to a layer hub to keep L1/L2/L3 traversable.
        if not target_norm:
            target_norm = self._LAYER_HUBS.get(layer_norm, "__hub_l3__")
            if target_norm == entity_norm:
                target_norm = "__hub_shared__"
        if self._is_duplicate_graph_fact(entity_norm, fact, threshold=0.70):
            return {"ok": True, "layer": layer_norm, "fact": fact, "note": "duplicate_skipped"}
        self.add(
            "memory",
            fact,
            {
                "type": "fact",
                "layer": layer_norm,
                "relation": relation_norm,
                "entity": entity_norm,
                "target_entity": target_norm,
            },
        )
        graph_row = {
            "ts": utc_now(),
            "layer": layer_norm,
            "triple": {
                "entity": entity_norm,
                "relation": relation_norm,
                "target_entity": target_norm,
                "fact": fact,
            },
        }
        with self.graph_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(graph_row, ensure_ascii=False) + "\n")
        return {"ok": True, "layer": layer_norm, "fact": fact}

    def _load_graph(self) -> dict[str, list[dict[str, Any]]]:
        graph: dict[str, list[dict[str, Any]]] = {}
        if not self.graph_path.exists():
            return graph
        for raw in self.graph_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(raw)
                triple = row.get("triple", {})
                ent = triple.get("entity", "")
                if ent:
                    graph.setdefault(ent, []).append(row)
                target = triple.get("target_entity", "")
                if target:
                    graph.setdefault(target, []).append(row)
            except Exception:
                continue
        return graph

    @staticmethod
    def _word_tokens(text: str) -> set[str]:
        raw = text.lower().replace("-", " ").replace("_", " ").replace("/", " ")
        return {t for t in raw.split() if len(t) >= 2}

    def _extract_query_entities(self, query: str, graph_keys: list[str]) -> list[str]:
        query_lower = query.lower()
        query_words = self._word_tokens(query)
        matched: list[tuple[float, str]] = []
        for node in graph_keys:
            if not node:
                continue
            node_words = self._word_tokens(node)
            if not node_words:
                continue
            overlap = node_words & query_words
            if not overlap:
                if len(node) >= 2 and node.lower() in query_lower:
                    matched.append((0.3, node))
                continue
            coverage = len(overlap) / len(node_words)
            if coverage >= 0.5:
                matched.append((coverage, node))
        matched.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in matched]

    def recall_layers(self, query: str, per_layer: int = 3) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {"L1": [], "L2": [], "L3": []}
        if not self.graph_path.exists() and not self.path.exists():
            return out
        query_tokens = self._tokenize(query)
        graph = self._load_graph()
        for edge in graph.get("agent", []):
            layer = edge.get("layer", "L3")
            triple = edge.get("triple", {})
            rel = triple.get("relation", "")
            if layer == "L1" and rel in self._L1_RELATIONS:
                fact = triple.get("fact", "")
                if fact:
                    out["L1"].append({"content": fact, "meta": {"layer": "L1", "score": 1.0}})
        entry_nodes = self._extract_query_entities(query, list(graph.keys()))
        if not entry_nodes:
            return self._fallback_token_search(query_tokens, graph, out, per_layer)
        visited_nodes: set[str] = set()
        collected_facts: dict[str, tuple[float, str, str]] = {}
        queue: list[tuple[str, int]] = [(node, 0) for node in entry_nodes]
        hop_decay = {0: 1.0, 1: 0.6, 2: 0.3}
        while queue:
            node, depth = queue.pop(0)
            if node in visited_nodes or depth > 2:
                continue
            visited_nodes.add(node)
            for edge in graph.get(node, []):
                triple = edge.get("triple", {})
                fact = triple.get("fact", "")
                layer = edge.get("layer", "L3")
                rel = triple.get("relation", "")
                if layer == "L1" and rel in self._L1_RELATIONS:
                    continue
                if not fact:
                    continue
                fact_tokens = self._tokenize(fact)
                jaccard_sim = self._jaccard(query_tokens, fact_tokens)
                direct_hits = sum(1 for t in query_tokens if t in fact.lower()) / max(len(query_tokens), 1)
                raw_score = (jaccard_sim * 0.6 + direct_hits * 0.4) * hop_decay.get(depth, 0.1)
                if fact in collected_facts:
                    if raw_score > collected_facts[fact][0]:
                        collected_facts[fact] = (raw_score, fact, layer)
                else:
                    collected_facts[fact] = (raw_score, fact, layer)
                ent = triple.get("entity", "")
                tgt = triple.get("target_entity", "")
                if ent and ent not in visited_nodes:
                    queue.append((ent, depth + 1))
                if tgt and tgt not in visited_nodes:
                    queue.append((tgt, depth + 1))
        min_threshold = 0.05
        scored_list = sorted(collected_facts.values(), key=lambda x: x[0], reverse=True)
        for score, fact, layer in scored_list:
            if score < min_threshold:
                continue
            target_layer = layer if layer in {"L1", "L2", "L3"} else "L3"
            if target_layer == "L1":
                target_layer = "L2"
            if len(out[target_layer]) < per_layer:
                out[target_layer].append({"content": fact, "meta": {"layer": target_layer, "score": round(score, 3)}})
        return out

    def _fallback_token_search(
        self, query_tokens: set[str], graph: dict[str, list[dict[str, Any]]],
        out: dict[str, list[dict[str, Any]]], per_layer: int
    ) -> dict[str, list[dict[str, Any]]]:
        scored: list[tuple[float, str, str]] = []
        seen_facts: set[str] = set()
        for node_edges in graph.values():
            for edge in node_edges:
                triple = edge.get("triple", {})
                fact = triple.get("fact", "")
                layer = edge.get("layer", "L3")
                rel = triple.get("relation", "")
                if not fact or fact in seen_facts:
                    continue
                seen_facts.add(fact)
                if layer == "L1" and rel in self._L1_RELATIONS:
                    continue
                fact_tokens = self._tokenize(fact)
                sim = self._jaccard(query_tokens, fact_tokens)
                if sim > 0.08:
                    scored.append((sim, fact, layer))
        scored.sort(key=lambda x: x[0], reverse=True)
        for score, fact, layer in scored:
            target_layer = layer if layer in {"L2", "L3"} else "L3"
            if len(out[target_layer]) < per_layer:
                out[target_layer].append({"content": fact, "meta": {"layer": target_layer, "score": round(score, 3)}})
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: WorkspaceTools
# ═══════════════════════════════════════════════════════════════════════════════

class WorkspaceTools:
    def __init__(self, workspace: Path, memory: MemoryStore, brave_api_key: str = "", secret_store: SecretStore | None = None):
        self.workspace = workspace.resolve()
        self.memory = memory
        self.skills_dir = self.workspace / "skills"
        self.brave_api_key = brave_api_key
        self._secret_store = secret_store
        self._secrets = secret_store

    def _resolve_path(self, rel: str) -> Path:
        target = (self.workspace / rel).resolve()
        if self.workspace not in target.parents and target != self.workspace:
            raise ValueError("path is outside workspace")
        return target

    def get_all_tools(self) -> list[dict[str, Any]]:
        tools = list(TOOL_SPECS)
        if not self.skills_dir.exists():
            return tools
        for skill_dir in self.skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue
            manifest_file = skill_dir / "manifest.json"
            if not manifest_file.exists():
                continue
            try:
                spec = json.loads(manifest_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": str(spec.get("name") or skill_dir.name),
                        "description": str(spec.get("description") or ""),
                        "parameters": spec.get("parameters") or {"type": "object", "properties": {}},
                    },
                }
            )
        return tools

    def run_dynamic_tool(self, skill_name: str, args: dict[str, Any]) -> dict[str, Any]:
        skill_dir = (self.skills_dir / skill_name).resolve()
        if self.workspace not in skill_dir.parents:
            return {"ok": False, "error": "invalid skill path"}
        if not skill_dir.exists():
            return {"ok": False, "error": "skill not found"}
        arg_json = shlex.quote(safe_json_dumps(args))
        call_tool_script = Path("/home/jee/agent/planaria_one/call_external_tool.sh")
        if not call_tool_script.exists():
            legacy_script = Path("/home/jee/agent/planaria_one/call_tool.sh")
            if legacy_script.exists():
                call_tool_script = legacy_script
            else:
                return {"ok": False, "error": "global call_external_tool.sh script not found"}
        cmd = f"bash {shlex.quote(str(call_tool_script))} -w {shlex.quote(str(self.workspace))} -tool {shlex.quote(skill_name)} {arg_json}"
        return self.run_shell(cmd, timeout_sec=120)

    def update_config(self, section: str, key: str, value: str) -> dict[str, Any]:
        cfg_path = self.workspace / "config.json"
        data = safe_json_load(cfg_path, {})
        if section not in data:
            data[section] = {}
        # Store sensitive keys via SecretStore
        sensitive_keys = {"api_key", "brave_api_key", "token", "password", "app_password"}
        if key in sensitive_keys and self._secret_store:
            secret_name = f"{section}_{key}"
            self._secret_store.store(secret_name, value)
            data[section][key] = SecretStore.make_ref(secret_name)
        else:
            data[section][key] = value
        write_json(cfg_path, data)
        # Apply runtime changes immediately
        if section == "search" and key == "brave_api_key":
            self.brave_api_key = value
        return {"ok": True, "message": f"Config updated: {section}.{key} has been saved successfully."}

    def search_hub(self, query: str) -> dict[str, Any]:
        """Search GitHub for installable skills. Returns repos with compatibility info."""
        results: list[dict[str, Any]] = []
        errors: list[str] = []

        # Strategy 1: direct query + skill keywords
        search_queries = [
            f"{query} skill run.py manifest.json in:readme,name,description",
            f"{query} planaria skill in:readme,name,description",
            f"{query} in:readme,name,description language:python",
        ]
        seen_repos: set[str] = set()

        for sq in search_queries:
            if len(results) >= 8:
                break
            url = "https://api.github.com/search/repositories"
            params = {"q": sq, "sort": "stars", "order": "desc", "per_page": 5}
            try:
                resp = requests.get(url, params=params, headers={"User-Agent": "PLANARIA-Agent"}, timeout=15)
                if resp.status_code != 200:
                    errors.append(f"GitHub search error: {resp.status_code}")
                    continue
                for it in resp.json().get("items", []):
                    full_name = it.get("full_name", "")
                    if full_name in seen_repos:
                        continue
                    seen_repos.add(full_name)
                    results.append({
                        "name": full_name,
                        "url": it.get("html_url"),
                        "desc": it.get("description") or "",
                        "stars": it.get("stargazers_count", 0),
                        "clone_url": it.get("clone_url"),
                        "install_cmd": f'install_skill(repo_url="{it.get("clone_url")}", skill_name="{full_name.split("/")[-1]}")',
                    })
            except Exception as e:
                errors.append(str(e))

        if not results and errors:
            return {"ok": False, "error": "; ".join(errors)}
        return {
            "ok": True,
            "results": results[:8],
            "hint": "To install: call install_skill with the repo clone_url. install_skill will auto-validate structure and install dependencies.",
        }

    _NEWS_HINTS = {"뉴스", "동향", "현황", "전망", "근황", "시세", "주가", "증시", "환율", "금리",
                   "경제", "물가", "선거", "대선", "전쟁", "분쟁", "사건", "사고", "속보",
                   "news", "breaking", "market", "stock", "economy", "election", "war"}
    _KR_NEWS_SITES = "site:naver.com OR site:daum.net OR site:chosun.com OR site:hankyung.com OR site:mk.co.kr OR site:news.google.com"
    _BLOG_DOMAINS = {"blog.naver.com", "m.blog.naver.com", "brunch.co.kr", "tistory.com",
                     "velog.io", "medium.com", "blog.daum.net", "wordpress.com"}
    _RRF_K = 60
    _MAX_REWRITE_QUERIES = 4
    _MAX_REWRITE_QUERIES_COVERAGE = 6
    _MODEL_FAMILY_TERMS = [
        "gpt", "o1", "o3", "o4", "claude", "gemini", "llama", "qwen",
        "gemma", "deepseek", "mistral", "grok", "command r", "phi",
    ]

    @staticmethod
    def _detect_search_type(query: str) -> str:
        low = query.lower()
        news_score = sum(1 for h in WorkspaceTools._NEWS_HINTS if h in low)
        return "news" if news_score >= 1 else "knowledge"

    @staticmethod
    def _is_korean_query(query: str) -> bool:
        return bool(re.search(r"[\uac00-\ud7a3]", query))

    @staticmethod
    def _normalize_query_text(query: str) -> str:
        return " ".join((query or "").strip().split())

    @staticmethod
    def _is_model_landscape_query(query: str) -> bool:
        low = (query or "").lower()
        if not low:
            return False
        topic_hit = any(k in low for k in ["llm", "모델", "model", "언어모델", "language model"])
        enum_hit = any(k in low for k in ["최신", "latest", "비교", "목록", "리스트", "top", "추천", "종류"])
        return topic_hit and enum_hit

    @classmethod
    def _extract_model_family_hits(cls, text: str) -> set[str]:
        low = (text or "").lower()
        out: set[str] = set()
        for term in cls._MODEL_FAMILY_TERMS:
            if term in low:
                out.add(term)
        return out

    @classmethod
    def _inject_coverage_queries(cls, rewrites: list[str], query: str, search_type: str, is_ko: bool) -> list[str]:
        if search_type != "knowledge":
            return rewrites
        if not cls._is_model_landscape_query(query):
            return rewrites
        coverage_templates = [
            "latest LLM models GPT Claude Gemini Llama Qwen Gemma DeepSeek Mistral",
            "LLM leaderboard GPT Claude Gemini Llama Qwen Gemma DeepSeek Mistral",
        ]
        if is_ko:
            coverage_templates = [
                "최신 LLM 모델 GPT Claude Gemini Llama Qwen Gemma DeepSeek Mistral 비교",
                "LLM 리더보드 GPT Claude Gemini Llama Qwen Gemma DeepSeek Mistral",
            ]
        for c in coverage_templates:
            if c not in rewrites:
                rewrites.append(c)
            if len(rewrites) >= cls._MAX_REWRITE_QUERIES_COVERAGE:
                break
        return rewrites

    def _build_rewrite_queries(self, query: str, search_type: str) -> list[str]:
        base = self._normalize_query_text(query)
        if not base:
            return []
        rewrites: list[str] = [base]
        is_ko = self._is_korean_query(base)
        if search_type == "news":
            suffixes = ["최신", "오늘", "속보"] if is_ko else ["latest", "today", "breaking"]
        else:
            suffixes = ["핵심 정리", "개요", "공식 문서"] if is_ko else ["overview", "official docs", "guide"]
        for suffix in suffixes:
            q = f"{base} {suffix}".strip()
            if q not in rewrites:
                rewrites.append(q)
            if len(rewrites) >= self._MAX_REWRITE_QUERIES:
                break
        rewrites = self._inject_coverage_queries(rewrites, base, search_type, is_ko)
        max_q = self._MAX_REWRITE_QUERIES_COVERAGE if self._is_model_landscape_query(base) else self._MAX_REWRITE_QUERIES
        return rewrites[:max_q]

    @staticmethod
    def _result_dedupe_key(row: dict[str, Any]) -> str:
        url = str(row.get("url") or "").strip().lower()
        if url:
            return f"url:{url}"
        title = str(row.get("title") or "").strip().lower()
        snippet = str(row.get("snippet") or "").strip().lower()
        return f"text:{title}|{snippet}"

    def _search_via_brave(self, query: str, search_type: str, is_ko: bool) -> tuple[list[dict[str, Any]] | None, str]:
        """Brave Search call. Returns (results, error_msg). results=None on failure."""
        if not self.brave_api_key:
            return None, "Brave API key not configured"
        headers = {"Accept": "application/json", "X-Subscription-Token": self.brave_api_key}
        if search_type == "news":
            url = "https://api.search.brave.com/res/v1/news/search"
            params = {"q": query, "count": 10, "freshness": "pw", "spellcheck": 1, "text_decorations": 0}
        else:
            url = "https://api.search.brave.com/res/v1/web/search"
            params = {"q": query, "count": 10, "extra_snippets": 1, "text_decorations": 0, "spellcheck": 1}
        if is_ko:
            params.update({"country": "kr", "search_lang": "ko"})
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            if resp.status_code != 200:
                return None, f"Brave API error: {resp.status_code} {resp.text[:200]}"
            data = resp.json()
            raw_results = data.get("results", []) if search_type == "news" else data.get("web", {}).get("results", [])
            results: list[dict[str, Any]] = []
            for item in raw_results:
                item_url = item.get("url", "")
                source = (item.get("meta_url") or {}).get("netloc", "")
                if search_type == "news":
                    domain = source or item_url
                    if any(bd in domain for bd in self._BLOG_DOMAINS):
                        continue
                entry = {"title": item.get("title", ""), "snippet": item.get("description", ""), "url": item_url}
                age = item.get("age") or item.get("page_age") or ""
                if age:
                    entry["age"] = age
                if source:
                    entry["source"] = source
                extras = item.get("extra_snippets", [])
                if extras:
                    entry["extra_snippets"] = extras
                results.append(entry)
            return results, ""
        except Exception as e:
            return None, f"Brave request failed: {e}"

    def _search_via_duckduckgo(self, query: str) -> tuple[list[dict[str, Any]] | None, str]:
        """DuckDuckGo Search call. Returns (results, error_msg). results=None on failure.

        Prefer the new `ddgs` package; fall back to legacy `duckduckgo_search`.
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    from ddgs import DDGS  # type: ignore
                except ImportError:
                    from duckduckgo_search import DDGS  # legacy
                with DDGS() as ddgs:
                    ddg_results = list(ddgs.text(query, max_results=10))
            results = [
                {"title": item.get("title", ""), "snippet": item.get("body", ""), "url": item.get("href", "")}
                for item in ddg_results
            ]
            return results, ""
        except ImportError:
            return None, "duckduckgo-search/ddgs package not installed (uv add ddgs)"
        except Exception as e:
            return None, f"DuckDuckGo request failed: {e}"

    def _single_web_search(self, query: str, search_type: str) -> dict[str, Any]:
        """Web search: Brave when configured (higher quality), DuckDuckGo as automatic fallback.

        DDG fallback requires no API key and should always be available out of the box.
        """
        is_ko = self._is_korean_query(query)

        # 1) Primary: Brave (when API key configured)
        brave_results, brave_err = self._search_via_brave(query, search_type, is_ko)
        if brave_results is not None and len(brave_results) > 0:
            return {"ok": True, "results": brave_results, "source": f"brave_{search_type}", "search_type": search_type}

        # 2) Fallback: DuckDuckGo (always-available, no key required)
        ddg_results, ddg_err = self._search_via_duckduckgo(query)
        if ddg_results is not None:
            note = f"(Brave unavailable: {brave_err})" if brave_err else ""
            out = {"ok": True, "results": ddg_results, "source": "duckduckgo", "search_type": search_type}
            if note:
                out["note"] = note
            return out

        # 3) Both failed
        return {"ok": False, "error": f"Brave: {brave_err} | DuckDuckGo: {ddg_err}"}

    def _rrf_fuse_results(self, per_query_results: list[tuple[str, list[dict[str, Any]]]], limit: int = 10) -> list[dict[str, Any]]:
        score_map: dict[str, float] = {}
        row_map: dict[str, dict[str, Any]] = {}
        source_queries: dict[str, list[str]] = {}
        for q, rows in per_query_results:
            for rank, row in enumerate(rows, start=1):
                key = self._result_dedupe_key(row)
                if not key:
                    continue
                score_map[key] = score_map.get(key, 0.0) + (1.0 / float(self._RRF_K + rank))
                if key not in row_map:
                    row_map[key] = dict(row)
                source_queries.setdefault(key, [])
                if q not in source_queries[key]:
                    source_queries[key].append(q)
        fused: list[tuple[float, dict[str, Any], list[str]]] = []
        for key, score in score_map.items():
            fused.append((score, row_map.get(key, {}), source_queries.get(key, [])))
        fused.sort(key=lambda x: x[0], reverse=True)
        out: list[dict[str, Any]] = []
        for score, row, qs in fused[:limit]:
            merged = dict(row)
            merged["rrf_score"] = round(float(score), 6)
            merged["matched_queries"] = qs[:3]
            out.append(merged)
        return out

    @classmethod
    def _diversify_model_family_results(cls, rows: list[dict[str, Any]], query: str, limit: int = 10) -> list[dict[str, Any]]:
        if not cls._is_model_landscape_query(query):
            return rows[:limit]
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        seen_families: set[str] = set()
        for row in rows:
            text = f"{row.get('title', '')} {row.get('snippet', '')}"
            families = cls._extract_model_family_hits(text)
            if not families:
                continue
            if families.issubset(seen_families):
                continue
            key = str(row.get("url") or id(row))
            if key in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(key)
            seen_families.update(families)
            if len(selected) >= min(6, limit):
                break
        for row in rows:
            if len(selected) >= limit:
                break
            key = str(row.get("url") or id(row))
            if key in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(key)
        return selected[:limit]

    def web_search(self, query: str, search_type: str = "auto") -> dict[str, Any]:
        if search_type == "auto":
            search_type = self._detect_search_type(query)
        rewrites = self._build_rewrite_queries(query, search_type)
        if not rewrites:
            return {"ok": False, "error": "empty query"}
        successes: list[tuple[str, list[dict[str, Any]]]] = []
        errors: list[str] = []
        source_names: list[str] = []
        for q in rewrites:
            res = self._single_web_search(q, search_type)
            if res.get("ok"):
                rows = res.get("results", [])
                if isinstance(rows, list) and rows:
                    successes.append((q, rows))
                    src = str(res.get("source") or "")
                    if src and src not in source_names:
                        source_names.append(src)
                continue
            err = str(res.get("error") or "")
            if err:
                errors.append(err)
        if not successes:
            return {"ok": False, "error": errors[0] if errors else "search failed"}
        fused = self._rrf_fuse_results(successes, limit=20)
        fused = self._diversify_model_family_results(fused, query=query, limit=10)
        source_label = "rrf:" + ",".join(source_names) if source_names else "rrf:unknown"
        return {
            "ok": True,
            "results": fused,
            "source": source_label,
            "search_type": search_type,
            "rewrites": rewrites,
            "rewrite_count": len(rewrites),
            "fused_from": len(successes),
        }

    @staticmethod
    def _infer_recurrence_from_prompt(task_prompt: str) -> str:
        low = (task_prompt or "").lower()
        if ("매일" in task_prompt) or ("every day" in low) or ("daily" in low):
            return "daily"
        if ("매주" in task_prompt) or ("every week" in low) or ("weekly" in low):
            return "weekly"
        if ("매월" in task_prompt) or ("every month" in low) or ("monthly" in low):
            return "monthly"
        return "none"

    @staticmethod
    def _normalize_recurrence(recurrence: str, task_prompt: str) -> str:
        rec = (recurrence or "auto").strip().lower()
        if rec == "auto":
            rec = WorkspaceTools._infer_recurrence_from_prompt(task_prompt)
        if rec not in {"none", "daily", "weekly", "monthly"}:
            rec = "none"
        return rec

    def schedule_task(self, run_at_utc: str, task_prompt: str, recurrence: str = "auto", interval: int = 1) -> dict[str, Any]:
        try:
            target_time = datetime.fromisoformat(run_at_utc.replace('Z', '+00:00'))
            if target_time.tzinfo is None:
                target_time = target_time.replace(tzinfo=timezone.utc)
            else:
                target_time = target_time.astimezone(timezone.utc)
            task_file = self.workspace / "data" / "scheduled_tasks.jsonl"
            task_file.parent.mkdir(parents=True, exist_ok=True)
            rec = self._normalize_recurrence(recurrence, task_prompt)
            iv = max(1, min(int(interval or 1), 365))
            task_data = {
                "id": str(time.time()),
                "run_at": target_time.isoformat(),
                "prompt": task_prompt,
                "status": "pending",
                "recurrence": rec,
                "interval": iv,
                "created_at": utc_now(),
                "last_run_at": None,
            }
            with task_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(task_data, ensure_ascii=False) + "\n")
            return {
                "ok": True,
                "message": f"Task scheduled successfully for {run_at_utc}",
                "run_at_utc": target_time.isoformat().replace("+00:00", "Z"),
                "recurrence": rec,
                "interval": iv,
            }
        except ValueError:
            return {"ok": False, "error": "Invalid time format. Use ISO 8601 UTC (e.g., 2026-03-17T09:00:00Z)"}

    def list_scheduled_tasks(self, include_done: bool = False) -> dict[str, Any]:
        """List all scheduled tasks for this workspace.

        Reads scheduled_tasks.jsonl and returns a structured list. The agent
        MUST call this tool instead of making up task lists from memory.
        """
        task_file = self.workspace / "data" / "scheduled_tasks.jsonl"
        if not task_file.exists():
            return {"ok": True, "tasks": [], "count": 0, "message": "No scheduled tasks."}
        tasks: list[dict[str, Any]] = []
        try:
            for line in task_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not include_done and row.get("status") == "done":
                    continue
                tasks.append({
                    "task_id": str(row.get("id", "")),
                    "run_at_utc": str(row.get("run_at", "")),
                    "prompt": str(row.get("prompt", ""))[:200],
                    "status": str(row.get("status", "")),
                    "recurrence": str(row.get("recurrence", "none")),
                    "interval": int(row.get("interval", 1) or 1),
                    "last_run_at": row.get("last_run_at"),
                })
        except Exception as e:
            return {"ok": False, "error": f"failed to read tasks: {e}"}
        return {"ok": True, "tasks": tasks, "count": len(tasks)}

    def cancel_scheduled_task(self, task_id: str) -> dict[str, Any]:
        """Cancel a scheduled task by id. Removes it from the queue file."""
        task_file = self.workspace / "data" / "scheduled_tasks.jsonl"
        if not task_file.exists():
            return {"ok": False, "error": "no scheduled tasks file"}
        target_id = str(task_id).strip()
        if not target_id:
            return {"ok": False, "error": "empty task_id"}
        kept: list[str] = []
        removed = False
        for line in task_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                kept.append(line)
                continue
            if str(row.get("id", "")) == target_id:
                removed = True
                continue
            kept.append(json.dumps(row, ensure_ascii=False))
        if not removed:
            return {"ok": False, "error": f"task_id not found: {target_id}"}
        task_file.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        return {"ok": True, "cancelled_id": target_id, "remaining": len(kept)}

    def list_files(self, path: str = ".") -> dict[str, Any]:
        p = self._resolve_path(path)
        if not p.exists():
            return {"ok": False, "error": "path does not exist"}
        if p.is_file():
            return {"ok": True, "items": [str(p.relative_to(self.workspace))]}
        items = []
        for x in sorted(p.iterdir()):
            suffix = "/" if x.is_dir() else ""
            items.append(str(x.relative_to(self.workspace)) + suffix)
        return {"ok": True, "items": items}

    def read_file(self, path: str) -> dict[str, Any]:
        p = self._resolve_path(path)
        if not p.exists() or not p.is_file():
            return {"ok": False, "error": "file not found"}
        try:
            text = p.read_text(encoding="utf-8")
            return {"ok": True, "content": text[:40000], "truncated": len(text) > 40000}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        p = self._resolve_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"ok": True, "bytes": len(content.encode("utf-8")), "path": str(p.relative_to(self.workspace))}

    def search_memory(self, query: str, limit: int = 5) -> dict[str, Any]:
        hits = self.memory.search(query, limit=max(1, min(limit, 20)))
        return {"ok": True, "hits": hits}

    def run_shell(self, command: str, timeout_sec: int = 30) -> dict[str, Any]:
        timeout_sec = max(1, min(timeout_sec, 300))
        try:
            proc = subprocess.run(
                command, shell=True, cwd=str(self.workspace),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                encoding="utf-8", errors="replace",  # never produce surrogates
                timeout=timeout_sec,
            )
            return {"ok": True, "exit_code": proc.returncode, "stdout": proc.stdout[-12000:], "stderr": proc.stderr[-12000:]}
        except Exception as e:
            return {"ok": False, "error": f"shell execution failed: {str(e)}"}

    def install_skill(self, repo_url: str, skill_name: str | None = None) -> dict[str, Any]:
        """Clone a skill repo, validate structure, install deps, and register."""
        name = skill_name or repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        if not name:
            return {"ok": False, "error": "invalid skill name"}
        dest = (self.skills_dir / name).resolve()
        if dest.exists():
            return {"ok": False, "error": f"skill already exists: {name}. Use run_skill to execute it."}
        self.skills_dir.mkdir(parents=True, exist_ok=True)

        # 1. Clone
        try:
            proc = subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(dest)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60,
            )
            if proc.returncode != 0:
                return {"ok": False, "error": proc.stderr.strip() or "git clone failed"}
        except FileNotFoundError:
            return {"ok": False, "error": "Git is not installed on the system."}
        except subprocess.TimeoutExpired:
            shutil.rmtree(dest, ignore_errors=True)
            return {"ok": False, "error": "git clone timed out (60s)"}
        except Exception as e:
            shutil.rmtree(dest, ignore_errors=True)
            return {"ok": False, "error": str(e)}

        # 2. Validate structure — find run.py (may be in root or subdirectory)
        run_py = dest / "run.py"
        if not run_py.exists():
            # Check common patterns: src/run.py, skill/run.py, <name>/run.py
            for candidate in dest.rglob("run.py"):
                run_py = candidate
                break
        has_run_py = run_py.exists()
        has_run_sh = (dest / "run.sh").exists()
        has_manifest = (dest / "manifest.json").exists()

        # If run.py is in subdirectory, restructure
        if has_run_py and run_py.parent != dest:
            # Move contents of subdirectory up
            sub_dir = run_py.parent
            for item in sub_dir.iterdir():
                target = dest / item.name
                if not target.exists():
                    item.rename(target)
            has_run_py = (dest / "run.py").exists()
            has_manifest = (dest / "manifest.json").exists()

        # 3. Auto-generate manifest if missing but run.py exists
        if has_run_py and not has_manifest:
            # Read first few lines for docstring
            try:
                code_text = (dest / "run.py").read_text(encoding="utf-8")
                desc = ""
                for line in code_text.split("\n")[:10]:
                    stripped = line.strip().strip('"').strip("'")
                    if stripped and not stripped.startswith("#!") and not stripped.startswith("import"):
                        desc = stripped[:200]
                        break
            except Exception:
                desc = ""
            manifest = {"name": name, "description": desc or f"Skill installed from {repo_url}", "parameters": {"type": "object", "properties": {}}}
            (dest / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            has_manifest = True

        # 4. Install dependencies
        install_log = ""
        for req_file in ["requirements.txt", "requirements.in"]:
            req_path = dest / req_file
            if req_path.exists():
                pkgs = [l.strip() for l in req_path.read_text().splitlines() if l.strip() and not l.startswith("#")]
                if pkgs:
                    cmd = f"uv pip install {' '.join(pkgs)}"
                    res = self.run_shell(cmd, timeout_sec=120)
                    install_log = f"[deps: {len(pkgs)} packages, exit={res.get('exit_code')}]"
                break

        # 5. Clean up .git to save space
        git_dir = dest / ".git"
        if git_dir.exists():
            shutil.rmtree(git_dir, ignore_errors=True)

        # 6. Report
        status_parts = []
        if has_run_py:
            status_parts.append("run.py ✓")
        elif has_run_sh:
            status_parts.append("run.sh ✓")
        else:
            status_parts.append("WARNING: no run.py or run.sh found")
        if has_manifest:
            status_parts.append("manifest.json ✓")
        if install_log:
            status_parts.append(install_log)

        # Read manifest for tool info
        tool_info = ""
        if has_manifest:
            try:
                m = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
                tool_info = f"Registered as tool '{m.get('name', name)}'. "
            except Exception:
                pass

        return {
            "ok": True,
            "skill": name,
            "path": str(dest.relative_to(self.workspace)),
            "structure": ", ".join(status_parts),
            "note": f"{tool_info}Use run_skill(skill_name='{name}', args='...') or call '{name}' directly as a tool. {install_log}",
        }

    def run_skill(self, skill_name: str, args: str = "") -> dict[str, Any]:
        skill_dir = (self.skills_dir / skill_name).resolve()
        if self.workspace not in skill_dir.parents:
            return {"ok": False, "error": "invalid skill path"}
        if not skill_dir.exists():
            return {"ok": False, "error": "skill not found"}
        run_sh = skill_dir / "run.sh"
        run_py = skill_dir / "run.py"
        if run_sh.exists():
            cmd = f"bash {shlex.quote(str(run_sh))} {args}".strip()
        elif run_py.exists():
            cmd = f"python3 {shlex.quote(str(run_py))} {args}".strip()
        else:
            return {"ok": False, "error": "skill has no run.sh or run.py"}
        return self.run_shell(cmd, timeout_sec=120)

    # ── Email & Calendar integration ─────────────────────────────

    def check_email(self, provider: str, limit: int = 5, unread_only: bool = True) -> dict[str, Any]:
        """Check email inbox via IMAP (Gmail)."""
        cfg = safe_json_load(self.workspace / "config.json", {})
        integrations = cfg.get("integrations", {})
        if provider == "gmail":
            ecfg = integrations.get("gmail", {})
        else:
            return {"ok": False, "error": f"Unknown provider: {provider}. Use 'gmail'."}
        if not ecfg.get("enabled"):
            return {"ok": False, "error": f"{provider} 이메일이 설정되어 있지 않습니다."}
        host = ecfg.get("imap_host", "")
        port = int(ecfg.get("imap_port", 993))
        username = ecfg.get("username", "")
        raw_pw = ecfg.get("app_password", "") or ecfg.get("password", "")
        password = (self._secrets.load(SecretStore.ref_key(raw_pw)) if self._secrets and SecretStore.is_ref(raw_pw) else raw_pw) or ""
        mailbox = ecfg.get("mailbox", "INBOX")
        use_ssl = ecfg.get("use_ssl", True)
        if not host or not username or not password:
            return {"ok": False, "error": f"{provider} 이메일 인증 정보가 불완전합니다."}
        try:
            mail = imaplib.IMAP4_SSL(host, port) if use_ssl else imaplib.IMAP4(host, port)
            mail.login(username, password)
            mail.select(mailbox)
            criteria = "UNSEEN" if unread_only else "ALL"
            _status, msg_ids = mail.search(None, criteria)
            ids = msg_ids[0].split()
            ids = ids[-limit:]
            results: list[dict[str, str]] = []
            for mid in reversed(ids):
                _status, msg_data = mail.fetch(mid, "(RFC822)")
                raw = msg_data[0][1]
                msg = email_mod.message_from_bytes(raw)
                results.append({
                    "subject": _decode_email_header(msg["Subject"]),
                    "from": _decode_email_header(msg["From"]),
                    "date": msg["Date"] or "",
                    "snippet": _extract_email_body(msg)[:500],
                })
            mail.logout()
            return {"ok": True, "provider": provider, "count": len(results), "emails": results}
        except Exception as e:
            return {"ok": False, "error": f"IMAP error: {type(e).__name__}: {safe_str(str(e))}"}

    def check_calendar(self, days_ahead: int = 1, max_results: int = 10) -> dict[str, Any]:
        """Check Google Calendar for upcoming events."""
        cfg = safe_json_load(self.workspace / "config.json", {})
        gcal_cfg = cfg.get("integrations", {}).get("google_calendar", {})
        if not gcal_cfg.get("enabled"):
            return {"ok": False, "error": "Google Calendar가 설정되어 있지 않습니다."}
        creds_path = gcal_cfg.get("credentials_json_path", "")
        token_path = gcal_cfg.get("token_json_path", "") or str(self.workspace / "data" / "gcal_token.json")
        calendar_id = gcal_cfg.get("calendar_id", "primary")
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request as GRequest
            from googleapiclient.discovery import build
        except ImportError:
            return {"ok": False, "error": "Google Calendar 패키지가 필요합니다. 실행: uv pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"}
        scopes = ["https://www.googleapis.com/auth/calendar.readonly"]
        creds = None
        if Path(token_path).exists():
            creds = Credentials.from_authorized_user_file(token_path, scopes)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(GRequest())
            else:
                if not creds_path or not Path(creds_path).exists():
                    return {"ok": False, "error": "Google Calendar credentials.json 파일을 찾을 수 없습니다."}
                flow = InstalledAppFlow.from_client_secrets_file(creds_path, scopes)
                creds = flow.run_local_server(port=0)
            Path(token_path).parent.mkdir(parents=True, exist_ok=True)
            Path(token_path).write_text(creds.to_json())
        service = build("calendar", "v3", credentials=creds)
        now_dt = datetime.now(timezone.utc)
        end_dt = now_dt + timedelta(days=min(max(days_ahead, 1), 7))
        events_result = service.events().list(
            calendarId=calendar_id, timeMin=now_dt.isoformat(), timeMax=end_dt.isoformat(),
            maxResults=max_results, singleEvents=True, orderBy="startTime",
        ).execute()
        events: list[dict[str, str]] = []
        for ev in events_result.get("items", []):
            start = ev.get("start", {}).get("dateTime", ev.get("start", {}).get("date", ""))
            events.append({
                "summary": ev.get("summary", "(제목 없음)"),
                "start": start,
                "end": ev.get("end", {}).get("dateTime", ""),
                "location": ev.get("location", ""),
            })
        return {"ok": True, "count": len(events), "events": events}

    def remember_fact(self, fact: str, layer: str = "L3", relation: str = "related_to", entity: str = "user", target_entity: str = "") -> dict[str, Any]:
        return self.memory.remember_fact(fact=fact, layer=layer, relation=relation, entity=entity, target_entity=target_entity)

    def reset_workspace(self, confirm: str) -> dict[str, Any]:
        if confirm != "RESET":
            return {"ok": False, "error": "reset aborted: confirm must be RESET"}
        removed: list[str] = []
        for rel in ["memory/memory.jsonl", "memory/graph_memory.jsonl", "logs/runtime_log.jsonl", "data/scheduled_tasks.jsonl"]:
            p = self._resolve_path(rel)
            if p.exists():
                p.unlink()
                removed.append(rel)
        for rel_dir in ["skills"]:
            d = self._resolve_path(rel_dir)
            if d.exists() and d.is_dir():
                for child in d.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink(missing_ok=True)
                removed.append(rel_dir + "/*")
        return {"ok": True, "removed": removed}


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: LLMClient
# ═══════════════════════════════════════════════════════════════════════════════

class LLMClient:
    def __init__(self, api_base: str, api_key: str, models: list[str], timeout_sec: int,
                 fallback_routes: list[dict[str, str]] | None = None,
                 cooldown_sec: int = DEFAULT_ROUTE_COOLDOWN_SEC,
                 state_path: Path | None = None):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.models = models
        self.timeout_sec = timeout_sec
        self.fallback_routes = fallback_routes or []
        self.cooldown_sec = max(60, int(cooldown_sec))
        self.state_path = state_path
        self.route_cooldowns: dict[str, float] = {}
        self.route_timeout_overrides: dict[str, float] = {}
        self._load_route_state()

    def _append_llm_trace(self, row: dict[str, Any]) -> None:
        if not self.state_path:
            return
        try:
            trace_path = self.state_path.parent / "llm_trace.jsonl"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _chat_url(self, api_base: str, model: str) -> str:
        base = api_base.rstrip("/")
        model_low = (model or "").lower()
        if "deepseek" in model_low and base == "https://api.deepseek.com":
            base = base + "/v1"
        return f"{base}/chat/completions"

    def _route_key(self, model: str, api_base: str) -> str:
        return f"{api_base.rstrip('/')}/{model}"

    def _load_route_state(self) -> None:
        if not self.state_path or not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            rows = raw.get("cooldowns", {}) if isinstance(raw, dict) else {}
            if isinstance(rows, dict):
                self.route_cooldowns = {str(k): float(v) for k, v in rows.items()}
                self._prune_expired_cooldowns()
            ovs = raw.get("timeout_overrides", {}) if isinstance(raw, dict) else {}
            if isinstance(ovs, dict):
                self.route_timeout_overrides = {str(k): float(v) for k, v in ovs.items()}
        except Exception:
            self.route_cooldowns = {}
            self.route_timeout_overrides = {}

    def _prune_expired_cooldowns(self) -> None:
        now = time.time()
        self.route_cooldowns = {k: v for k, v in self.route_cooldowns.items() if float(v) > now}

    def _save_route_state(self) -> None:
        if not self.state_path:
            return
        try:
            self._prune_expired_cooldowns()
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"cooldowns": self.route_cooldowns, "timeout_overrides": self.route_timeout_overrides, "updated_at": utc_now()}
            self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _is_cooldown_active(self, route_key: str) -> tuple[bool, float]:
        until = float(self.route_cooldowns.get(route_key, 0.0))
        return (time.time() < until, until)

    def _mark_timeout_cooldown(self, route_key: str, reason: str = "timeout") -> None:
        until = time.time() + float(self.cooldown_sec)
        self.route_cooldowns[route_key] = until
        left = int(max(0, until - time.time()))
        print(f"[LLMClient cooldown] mark route={route_key} for {left}s reason={reason}")
        self._save_route_state()

    def _effective_timeout(self, route_key: str, configured: float, is_last_route: bool) -> float:
        base = min(float(LLM_QUERY_TIMEOUT_SEC), configured)
        if not is_last_route:
            return base
        ov = float(self.route_timeout_overrides.get(route_key, base))
        return max(base, min(float(LLM_QUERY_TIMEOUT_MAX_SEC), ov))

    def _increase_last_route_timeout(self, route_key: str, configured: float) -> None:
        base = min(float(LLM_QUERY_TIMEOUT_SEC), configured)
        old = float(self.route_timeout_overrides.get(route_key, base))
        new = min(float(LLM_QUERY_TIMEOUT_MAX_SEC), old + float(LLM_QUERY_TIMEOUT_STEP_SEC))
        if new > old:
            self.route_timeout_overrides[route_key] = new
            print(f"[LLMClient timeout-adapt] route={route_key} timeout {old:.0f}s -> {new:.0f}s")
            self._save_route_state()

    def _is_timeout_like_exception(self, err: Exception) -> bool:
        if isinstance(err, requests.exceptions.Timeout):
            return True
        text = str(err).lower()
        return ("timed out" in text) or ("read timeout" in text) or ("timeout" in text)

    def _normalize_value(self, value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, dict):
            return {str(k): self._normalize_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._normalize_value(v) for v in value]
        if isinstance(value, str):
            return safe_str(value)
        return value

    def _normalize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for msg in messages:
            role = str(msg.get("role") or "user")
            content = msg.get("content")
            if content is None:
                content = ""
            if not isinstance(content, str):
                content = json.dumps(self._normalize_value(content), ensure_ascii=False)
            row: dict[str, Any] = {"role": role, "content": safe_str(content)}
            if msg.get("tool_calls") is not None:
                row["tool_calls"] = self._normalize_value(msg.get("tool_calls"))
            if msg.get("tool_call_id") is not None:
                row["tool_call_id"] = str(msg.get("tool_call_id"))
            normalized.append(row)
        return normalized

    def verify_key(self, api_base: str, api_key: str, model: str) -> tuple[bool, str]:
        """Quick API key verification. Returns (ok, message)."""
        url = self._chat_url(api_base.rstrip("/"), model)
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
            if resp.status_code == 200:
                return True, "ok"
            if resp.status_code == 401:
                return False, "인증 실패 — API 키가 올바르지 않습니다."
            if resp.status_code == 403:
                return False, "접근 거부 — API 키 권한을 확인해주세요."
            return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
        except requests.exceptions.Timeout:
            return True, "timeout (키는 유효할 수 있음)"
        except Exception as e:
            return False, f"연결 실패: {e}"

    def _post_with_cancel(
        self,
        url: str,
        headers: dict[str, str],
        payload_json: str,
        timeout: float,
        cancel_event: threading.Event | None = None,
    ) -> requests.Response:
        if cancel_event is None:
            return requests.post(url, headers=headers, data=payload_json, timeout=timeout)
        if cancel_event.is_set():
            raise RuntimeError("cancelled_by_client")

        session = requests.Session()
        done = threading.Event()
        outcome: dict[str, Any] = {}

        def _worker() -> None:
            try:
                outcome["resp"] = session.post(url, headers=headers, data=payload_json, timeout=timeout)
            except Exception as e:
                outcome["err"] = e
            finally:
                done.set()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        try:
            while True:
                if done.wait(timeout=0.1):
                    break
                if cancel_event.is_set():
                    try:
                        session.close()
                    except Exception:
                        pass
                    done.wait(timeout=1.0)
                    raise RuntimeError("cancelled_by_client")
            if "err" in outcome:
                raise outcome["err"]
            return outcome["resp"]
        finally:
            try:
                session.close()
            except Exception:
                pass

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        timeout_sec: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        has_primary = bool(self.api_key and self.api_base and self.models)
        has_fallback = bool(self.fallback_routes)
        if not has_primary and not has_fallback:
            raise RuntimeError("LLM API key is empty.")
        self._load_route_state()
        last_error = None
        attempted = 0
        routes: list[dict[str, str]] = []
        for model in self.models:
            routes.append({"model": model, "api_base": self.api_base, "api_key": self.api_key})
        routes.extend(self.fallback_routes)
        valid_routes: list[dict[str, str]] = []
        for route in routes:
            model = str(route.get("model") or "")
            route_base = str(route.get("api_base") or "").rstrip("/")
            route_key = str(route.get("api_key") or "")
            if model and route_base and route_key:
                valid_routes.append(route)
        for i, route in enumerate(valid_routes):
            model = str(route.get("model") or "")
            route_base = str(route.get("api_base") or "").rstrip("/")
            route_key = str(route.get("api_key") or "")
            rk = self._route_key(model, route_base)
            in_cd, until = self._is_cooldown_active(rk)
            if in_cd:
                left = int(max(0, until - time.time()))
                print(f"[LLMClient cooldown] skip {model} for {left}s")
                continue
            attempted += 1
            url = self._chat_url(route_base, model)
            payload: dict[str, Any] = {"model": model, "messages": self._normalize_messages(messages), "temperature": 0.2}
            if tools:
                payload["tools"] = self._normalize_value(tools)
                payload["tool_choice"] = "auto"
            headers = {"Authorization": f"Bearer {route_key}", "Content-Type": "application/json"}
            configured = self.timeout_sec if timeout_sec is None else max(1.0, float(timeout_sec))
            is_last_route = i == (len(valid_routes) - 1)
            timeout = self._effective_timeout(rk, configured, is_last_route=is_last_route)
            print(f"[LLMClient route] attempt model={model} base={route_base} t_budget={timeout:.1f}s")
            trace_base = {
                "ts": datetime.now(timezone.utc).isoformat(), "route_index": i, "route_total": len(valid_routes),
                "model": model, "api_base": route_base, "url": url, "timeout_sec": timeout, "request": payload,
            }
            try:
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("cancelled_by_client")
                resp = self._post_with_cancel(
                    url=url,
                    headers=headers,
                    payload_json=json.dumps(payload),
                    timeout=timeout,
                    cancel_event=cancel_event,
                )
                if resp.status_code >= 300:
                    last_error = RuntimeError(f"LLM request failed for model {model}: {resp.status_code} {resp.text[:500]}")
                    self._append_llm_trace({**trace_base, "status": "http_error", "http_status": resp.status_code, "response_text": safe_str(resp.text[:4000])})
                    print(f"[LLMClient fallback] {model} failed: {resp.status_code}")
                    continue
                print(f"[LLMClient route] success model={model} status={resp.status_code}")
                parsed = resp.json()
                self._append_llm_trace({**trace_base, "status": "ok", "http_status": resp.status_code, "response": parsed})
                return parsed
            except RuntimeError as e:
                if str(e) == "cancelled_by_client":
                    self._append_llm_trace({**trace_base, "status": "cancelled", "error_type": "RuntimeError", "error": safe_str(e)})
                    raise
                last_error = e
                self._append_llm_trace({**trace_base, "status": "exception", "error_type": type(e).__name__, "error": safe_str(e)})
                continue
            except requests.exceptions.Timeout as e:
                last_error = e
                self._append_llm_trace({**trace_base, "status": "exception", "error_type": type(e).__name__, "error": safe_str(e)})
                if is_last_route:
                    self._increase_last_route_timeout(rk, configured)
                    print(f"[LLMClient cooldown] skip mark for last route timeout route={rk}")
                else:
                    self._mark_timeout_cooldown(rk, reason=type(e).__name__)
                print(f"[LLMClient fallback] {model} triggered exception: {e}")
                continue
            except Exception as e:
                last_error = e
                self._append_llm_trace({**trace_base, "status": "exception", "error_type": type(e).__name__, "error": safe_str(e)})
                if self._is_timeout_like_exception(e):
                    if is_last_route:
                        self._increase_last_route_timeout(rk, configured)
                        print(f"[LLMClient cooldown] skip mark for last route timeout route={rk}")
                    else:
                        self._mark_timeout_cooldown(rk, reason=type(e).__name__)
                print(f"[LLMClient fallback] {model} triggered exception: {e}")
                continue
        if attempted == 0 and valid_routes:
            raise RuntimeError("All LLM routes are currently in cooldown.")
        raise last_error or RuntimeError("All models failed.")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: IntentClassifier (single source of truth for intent detection)
# ═══════════════════════════════════════════════════════════════════════════════

class IntentClassifier:
    """All keyword-based intent detection in one place. No state, all static methods."""

    # Intent hints for "what time is it now?" style queries. Deliberately
    # narrow — the fast-path returns a local-time string only, so any false
    # positive hijacks the turn. Tokens like "kst" or bare "현재 시간" match
    # mid-sentence in conversations about scheduling/news and must stay out
    # of this list.
    _LOCAL_TIME_HINTS = [
        "몇시", "몇 시", "지금 시간", "시간이 몇",
        "what time is it", "current time", "time is it now",
        "시스템 시간", "로컬 시간", "현재 시각", "local system time",
    ]
    _LOCAL_TIME_MAX_LEN = 30
    # Negations that mean "not about the current time" — when any of these
    # appears near the hint, skip the fast-path.
    _LOCAL_TIME_NEGATIONS = [
        "아니", "말고", "말구", "아니라", "아니에요",
        "not the", "not about",
    ]
    _MEMORY_STORE_HINTS = [
        "기억해", "기억해둬", "기억해줘", "잘 기억", "remember this", "remember that",
    ]
    _MEMORY_RECALL_HINTS = [
        "기억나", "기억 하고", "기억하고 있", "기억하는", "기억된", "기억해둔",
        "기억 중", "이전에", "예전에", "했었던 말", "했던 말", "뭐가 있었지",
        "뭐였는지", "뭐였지", "내 이름이 뭐였", "어디서 일한다고 했", "나열해",
        "remembered", "do you remember", "what did i say",
        "previously", "what do you know about me", "what have i told you",
    ]
    _WEB_GROUNDING_HINTS = [
        "현재", "지금", "오늘", "내일", "어제", "최근", "최신", "뉴스", "근황",
        "현황", "전망", "동향", "트렌드", "전쟁", "분쟁", "충돌", "외교", "정세",
        "주식", "시장", "증시", "환율", "금리", "경제", "물가", "날씨", "일정",
        "스케줄", "발표", "발사", "선거", "대선", "대통령", "논문", "연구", "이슈",
        "today", "now", "current", "latest", "news", "trend", "stock", "market",
        "economy", "election", "war", "conflict", "forecast", "paper", "research",
        "weather", "schedule",
    ]
    _CONTEXT_REASK_HINTS = [
        "이전 대화 기록", "이전 검색", "어떤 주제", "어떤 키워드",
        "검색을 도와드리기 전에", "알려주시겠", "무엇을 찾고", "주제를 알려",
        "provide", "previous search", "missing user input",
    ]
    _SIMPLE_DIALOG_ACTION_HINTS = [
        "파일", "폴더", "디렉터리", "검색", "요약", "분석", "저장", "생성", "작성",
        "수정", "삭제", "일정", "스케줄", "설정", "도구", "스킬", "명령", "코드",
        "테스트", "뉴스", "현황", "동향", "근황", "날씨", "주가", "전쟁", "예약",
        "이메일", "메일", "기억", "알려", "찾아", "확인", "조회",
        "file", "folder", "directory", "search", "summarize", "analyze",
        "save", "create", "write", "edit", "delete", "schedule", "config", "tool",
        "skill", "command", "code", "test", "news", "weather", "stock", "reserve",
        ".txt", ".py", ".md",
    ]
    _SIMPLE_DIALOG_GREETING_HINTS_KO = ["안녕", "반가", "넌 누구", "너 누구", "자기소개", "소개해", "고마워", "감사", "이름이 뭐", "너의 이름", "네 이름", "이름은?", "뭐라고 불러"]
    # Identity questions get matched BEFORE action hints to avoid
    # "이름 알려줘" being blocked by the "알려" action hint.
    _IDENTITY_HINTS = ["이름이 뭐", "너의 이름", "네 이름", "이름은?", "뭐라고 불러", "넌 누구", "너 누구"]
    _SUMMARY_HINTS = ["요약", "정리", "브리핑", "summary", "summarize"]
    # Narrow to direct, user-facing capability questions. Broad tokens like
    # "기능" match stray sentences mid-conversation (e.g. users asking about
    # external service features), so we require the phrasing to clearly refer
    # to the agent's own capabilities.
    _CAPABILITY_HINTS = [
        "뭘 시킬 수", "뭐를 시킬 수", "무엇을 할 수", "뭐 할 수",
        "뭐 할수", "뭘 할 수",
        "너 뭐 할", "너 뭘 할", "너가 할 수", "네가 할 수",
        "capabilities", "what can you do",
    ]
    _CAPABILITY_MAX_LEN = 50

    @staticmethod
    def is_local_time_query(user_text: str) -> bool:
        low = (user_text or "").strip().lower()
        if not low:
            return False
        # Long messages are almost certainly not a bare "what time is it".
        if len(low) > IntentClassifier._LOCAL_TIME_MAX_LEN:
            return False
        if any(neg in low for neg in IntentClassifier._LOCAL_TIME_NEGATIONS):
            return False
        return any(h in low for h in IntentClassifier._LOCAL_TIME_HINTS)

    @staticmethod
    def is_memory_store_query(user_text: str) -> bool:
        low = (user_text or "").strip().lower()
        return bool(low) and any(h in low for h in IntentClassifier._MEMORY_STORE_HINTS)

    @staticmethod
    def is_memory_recall_query(user_text: str) -> bool:
        low = (user_text or "").strip().lower()
        return bool(low) and any(h in low for h in IntentClassifier._MEMORY_RECALL_HINTS)

    @staticmethod
    def should_force_web_grounding(user_text: str) -> bool:
        low = (user_text or "").strip().lower()
        if not low or IntentClassifier.is_local_time_query(user_text):
            return False
        return any(h in low for h in IntentClassifier._WEB_GROUNDING_HINTS)

    @staticmethod
    def is_context_reask_text(text: str) -> bool:
        low = text.lower()
        if ("검색" not in text) and ("search" not in low):
            return False
        hit = sum(1 for h in IntentClassifier._CONTEXT_REASK_HINTS if h in low or h in text)
        return hit >= 2

    @staticmethod
    def is_simple_dialog_query(user_text: str) -> bool:
        low = (user_text or "").strip().lower()
        if not low:
            return True
        # Identity questions always fast-path (even if they contain action hint substrings like "알려")
        if any(h in low for h in IntentClassifier._IDENTITY_HINTS):
            return True
        if any(h in low for h in IntentClassifier._SIMPLE_DIALOG_ACTION_HINTS):
            return False
        if len(low) > 40:
            return False
        if any(h in low for h in IntentClassifier._SIMPLE_DIALOG_GREETING_HINTS_KO):
            return True
        if re.search(r"\b(hello|hi|thanks|thank you|who are you|introduce yourself)\b", low):
            return True
        return False

    @staticmethod
    def is_summary_request(user_text: str) -> bool:
        low = user_text.lower()
        return any(k in low for k in IntentClassifier._SUMMARY_HINTS)

    @staticmethod
    def is_capability_overview_query(user_text: str) -> bool:
        low = (user_text or "").strip().lower()
        if not low:
            return False
        if len(low) > IntentClassifier._CAPABILITY_MAX_LEN:
            return False
        return any(h in low for h in IntentClassifier._CAPABILITY_HINTS)

    @staticmethod
    def parse_tool_prohibitions(query: str, all_tool_names: list[str]) -> set[str]:
        low = query.lower()
        prohibition_markers = [
            "쓰지 말", "사용하지 말", "사용하지 마", "쓰지 마", "금지", "없이",
            "do not use", "don't use", "without using",
        ]
        prohibition_pos = -1
        for marker in prohibition_markers:
            idx = low.find(marker)
            if idx >= 0:
                prohibition_pos = idx
                break
        if prohibition_pos < 0:
            return set()
        prohibited_context = low[:prohibition_pos + 20]
        excluded: set[str] = set()
        alias_map: dict[str, list[str]] = {
            "exec": ["run_shell"], "shell": ["run_shell"], "셸": ["run_shell"],
            "쉘": ["run_shell"], "검색": ["web_search"], "웹": ["web_search"],
        }
        for alias, tools in alias_map.items():
            if alias in prohibited_context:
                excluded.update(tools)
        for name in all_tool_names:
            if name.lower() in prohibited_context:
                excluded.add(name)
        return excluded


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: ToolCallParser
# ═══════════════════════════════════════════════════════════════════════════════

class ToolCallParser:
    """Parse and normalize tool calls from LLM output. Pure functions, no state."""

    @staticmethod
    def extract_json_blocks(text: str) -> list[str]:
        blocks: list[str] = []
        stack: list[str] = []
        start_idx = -1
        in_string = False
        escape = False
        for i, char in enumerate(text):
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                if not stack:
                    start_idx = i
                stack.append(char)
            elif char == "}" and stack:
                stack.pop()
                if not stack and start_idx >= 0:
                    blocks.append(text[start_idx : i + 1])
                    start_idx = -1
        return blocks

    @staticmethod
    def extract_tagged_tool_calls(text: str) -> tuple[list[dict[str, Any]], str]:
        pattern = re.compile(
            r"<\|tool_call:begin\|>.*?<\|tool_call:name\|>(.*?)<\|tool_call:args\|>(\{.*?\})<\|tool_call:end\|>",
            re.DOTALL,
        )
        tool_calls: list[dict[str, Any]] = []
        cleaned = text
        for m in pattern.finditer(text):
            name = m.group(1).strip()
            args_raw = m.group(2).strip()
            try:
                args = json.loads(args_raw)
            except Exception:
                args = {}
            tool_calls.append({
                "id": f"call_tagged_{int(time.time()*1000)}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            })
            cleaned = cleaned.replace(m.group(0), "").strip()
        return tool_calls, cleaned

    @staticmethod
    def tool_signature(name: str, args: dict[str, Any]) -> str:
        try:
            arg_key = json.dumps(args, ensure_ascii=False, sort_keys=True)
        except Exception:
            arg_key = str(args)
        return f"{name}:{arg_key}"

    @staticmethod
    def normalize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize raw tool_calls into a consistent format. Returns at most 1 call."""
        normalized: list[dict[str, Any]] = []
        for i, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", {})
            if not isinstance(fn, dict):
                continue
            name = str(fn.get("name") or "").strip()
            if not name:
                continue
            raw_args = fn.get("arguments", "{}")
            if isinstance(raw_args, dict):
                args_str = json.dumps(raw_args, ensure_ascii=False)
            else:
                args_str = str(raw_args or "{}")
            tcid = str(tc.get("id") or "").strip()
            if not tcid:
                tcid = f"call_norm_{int(time.time()*1000)}_{i}"
            normalized.append({"id": tcid, "type": "function", "function": {"name": name, "arguments": args_str}})
        return normalized[:1]

    @staticmethod
    def parse_fallback_from_text(text: str) -> tuple[list[dict[str, Any]], str]:
        """Extract tool calls from plain text (tagged + JSON blocks). Returns (calls, cleaned_text)."""
        tool_calls: list[dict[str, Any]] = []
        cleaned_text = text

        tagged_calls, cleaned_text = ToolCallParser.extract_tagged_tool_calls(cleaned_text)
        tool_calls.extend(tagged_calls)

        json_blocks = ToolCallParser.extract_json_blocks(cleaned_text)
        for block in json_blocks:
            try:
                parsed = json.loads(block)
                target_func = None
                if "name" in parsed and "arguments" in parsed:
                    target_func = parsed
                elif "commentary" in parsed and isinstance(parsed["commentary"], dict) and "name" in parsed["commentary"]:
                    target_func = parsed["commentary"]
                if target_func:
                    args_str = json.dumps(target_func["arguments"], ensure_ascii=False) if isinstance(target_func["arguments"], dict) else str(target_func["arguments"])
                    tc = {
                        "id": f"call_fallback_{int(time.time()*1000)}",
                        "type": "function",
                        "function": {"name": str(target_func["name"]), "arguments": args_str},
                    }
                    tool_calls.append(tc)
                    cleaned_text = cleaned_text.replace(block, "").strip()
            except Exception:
                continue
        return tool_calls, cleaned_text


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8: ContextManager
# ═══════════════════════════════════════════════════════════════════════════════

class ContextManager:
    """Context window compression and sanitization.

    Bug fix: protects the most recent 2 user/assistant turns from compression,
    and prioritizes compressing tool results before conversation messages.
    """

    # Number of recent user+assistant messages to protect from compression
    PROTECTED_RECENT_TURNS = 4  # 2 user + 2 assistant

    def __init__(self, config: RuntimeConfig):
        self.config = config

    @staticmethod
    def messages_char_count(messages: list[dict[str, Any]]) -> int:
        total = 0
        for msg in messages:
            total += len(str(msg.get("role", "")))
            content = msg.get("content")
            if isinstance(content, str):
                total += len(content)
            total += len(json.dumps(msg.get("tool_calls", []), ensure_ascii=False))
        return total

    @staticmethod
    def shorten_tool_payloads(messages: list[dict[str, Any]], limit: int = 600) -> bool:
        changed = False
        for msg in messages:
            if msg.get("role") != "tool":
                continue
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            if len(content) > limit:
                msg["content"] = content[:limit] + "\n...[tool output compressed]"
                changed = True
        # Also compress system messages that contain tool results
        for msg in messages:
            if msg.get("role") != "system":
                continue
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            if content.startswith("Tool result (") and len(content) > limit:
                msg["content"] = content[:limit] + "\n...[tool result compressed]"
                changed = True
        return changed

    @staticmethod
    def truncate_latest_user_message(messages: list[dict[str, Any]], limit: int = 4000) -> bool:
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, str):
                return False
            if len(content) <= limit:
                return False
            head = content[:1200]
            tail = content[-800:]
            msg["content"] = head + "\n...[user input compressed]...\n" + tail + f"\n[original_length={len(content)}]"
            return True
        return False

    def compress_history_summary(self, messages: list[dict[str, Any]]) -> bool:
        if len(messages) < 8:
            return False
        keep_tail = self.config.max_recent_messages
        head = messages[:2]
        # Protect recent conversation turns from summarization
        protected_tail_count = max(keep_tail, self.PROTECTED_RECENT_TURNS)
        body = messages[2:-protected_tail_count] if len(messages) > (2 + protected_tail_count) else []
        tail = messages[-protected_tail_count:] if protected_tail_count > 0 else []
        summary_targets: list[dict[str, Any]] = []
        for msg in body:
            role = str(msg.get("role", ""))
            if role not in {"user", "assistant"}:
                continue
            content = str(msg.get("content", "")).strip()
            if content:
                summary_targets.append({"role": role, "content": content[:180]})
        if not summary_targets:
            return False
        lines = [f"{x['role']}: {x['content']}" for x in summary_targets[:18]]
        summary_msg = {"role": "system", "content": "Compressed conversation summary:\n" + "\n".join(lines)}
        messages[:] = head + [summary_msg] + tail
        return True

    @staticmethod
    def drop_oldest_non_system(messages: list[dict[str, Any]], protect_recent: int = 4) -> bool:
        """Drop the oldest non-system message, but protect the last `protect_recent` messages."""
        upper_bound = max(0, len(messages) - protect_recent)
        for i in range(upper_bound):
            if messages[i].get("role") == "system":
                continue
            del messages[i]
            return True
        return False

    @staticmethod
    def sanitize_orphan_tool_calls(messages: list[dict[str, Any]]) -> int:
        tool_ids: set[str] = set()
        for msg in messages:
            if msg.get("role") != "tool":
                continue
            tcid = str(msg.get("tool_call_id") or "").strip()
            if tcid:
                tool_ids.add(tcid)
        removed = 0
        cleaned_assistant: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") != "assistant":
                cleaned_assistant.append(msg)
                continue
            calls = msg.get("tool_calls")
            if not isinstance(calls, list) or not calls:
                cleaned_assistant.append(msg)
                continue
            kept_calls: list[dict[str, Any]] = []
            for tc in calls:
                tcid = str(tc.get("id") or "").strip() if isinstance(tc, dict) else ""
                if tcid and tcid in tool_ids:
                    kept_calls.append(tc)
            if len(kept_calls) != len(calls):
                removed += (len(calls) - len(kept_calls))
            if kept_calls:
                patched = dict(msg)
                patched["tool_calls"] = kept_calls
                cleaned_assistant.append(patched)
                continue
            content = str(msg.get("content") or "").strip()
            if content:
                patched = dict(msg)
                patched.pop("tool_calls", None)
                cleaned_assistant.append(patched)
            else:
                removed += 1
        valid_call_ids: set[str] = set()
        for msg in cleaned_assistant:
            if msg.get("role") != "assistant":
                continue
            calls = msg.get("tool_calls")
            if not isinstance(calls, list):
                continue
            for tc in calls:
                if not isinstance(tc, dict):
                    continue
                tcid = str(tc.get("id") or "").strip()
                if tcid:
                    valid_call_ids.add(tcid)
        cleaned_final: list[dict[str, Any]] = []
        for msg in cleaned_assistant:
            if msg.get("role") != "tool":
                cleaned_final.append(msg)
                continue
            tcid = str(msg.get("tool_call_id") or "").strip()
            if tcid and tcid in valid_call_ids:
                cleaned_final.append(msg)
            else:
                removed += 1
        if removed > 0:
            messages[:] = cleaned_final
        return removed

    def maybe_compress_context(self, messages: list[dict[str, Any]]) -> tuple[bool, list[str]]:
        notes: list[str] = []
        changed = False
        current = self.messages_char_count(messages)
        if current <= self.config.context_max_chars:
            return False, notes
        # Priority: compress tool results first, then user messages, then history
        if self.shorten_tool_payloads(messages):
            changed = True
            notes.append("tool_payloads_trimmed")
        current = self.messages_char_count(messages)
        if current > self.config.compressed_context_chars and self.truncate_latest_user_message(messages):
            changed = True
            notes.append("latest_user_truncated")
        current = self.messages_char_count(messages)
        if current > self.config.compressed_context_chars and self.compress_history_summary(messages):
            changed = True
            notes.append("history_summarized")
        current = self.messages_char_count(messages)
        while current > self.config.compressed_context_chars and self.drop_oldest_non_system(messages, protect_recent=self.PROTECTED_RECENT_TURNS):
            changed = True
            notes.append("dropped_oldest_message")
            current = self.messages_char_count(messages)
            if len(notes) > 64:
                break
        return changed, notes

    def aggressive_context_cut(self, messages: list[dict[str, Any]]) -> list[str]:
        notes: list[str] = []
        if self.shorten_tool_payloads(messages, limit=280):
            notes.append("aggressive_tool_trim")
        if self.truncate_latest_user_message(messages, limit=2200):
            notes.append("aggressive_user_trim")
        if self.compress_history_summary(messages):
            notes.append("aggressive_history_summary")
        while self.messages_char_count(messages) > max(3000, self.config.compressed_context_chars // 2):
            if not self.drop_oldest_non_system(messages, protect_recent=self.PROTECTED_RECENT_TURNS):
                break
            notes.append("aggressive_drop_oldest")
            if len(notes) > 128:
                break
        return notes


# ═══════════════════════════════════════════════════════════════════════════════
# Section 9: PromptBuilder
# ═══════════════════════════════════════════════════════════════════════════════

class PromptBuilder:
    """Assembles system prompts, task prompts, and execution context messages."""

    def __init__(self, config: RuntimeConfig, memory: MemoryStore):
        self.config = config
        self.memory = memory

    def build_system_prompt(self) -> str:
        name = self.config.agent_name
        aliases = self.config.aliases
        if aliases:
            alias_str = ", ".join(aliases)
            identity_line = f"You are {name} (also known as: {alias_str}), an autonomous assistant in PLANARIA_ONE."
        else:
            identity_line = f"You are {name}, an autonomous assistant in PLANARIA_ONE."

        # Dynamic workspace capabilities
        capabilities = []
        cfg = safe_json_load(self.config.workspace / "config.json", {})
        connectors = cfg.get("connectors", {})
        if connectors.get("srt_id"):
            capabilities.append("- SRT 열차 예약 가능 (srt_reserve 스킬 사용, 계정 설정 완료됨)")
        integrations = cfg.get("integrations", {})
        if integrations.get("gmail", {}).get("enabled"):
            capabilities.append("- Gmail 이메일 조회 가능")
        if integrations.get("google_calendar", {}).get("enabled"):
            capabilities.append("- Google Calendar 조회 가능")
        skills_dir = self.config.workspace / "skills"
        if skills_dir.is_dir():
            skill_names = [d.name for d in skills_dir.iterdir() if d.is_dir() and (d / "manifest.json").exists()]
            if skill_names:
                capabilities.append(f"- 설치된 스킬: {', '.join(skill_names)}")
        cap_block = "\n".join(capabilities) if capabilities else "- (없음)"

        today_str = datetime.now().strftime("%Y-%m-%d (%A)")
        return f"""{identity_line}

Today: {today_str}

Global principles:
1) Reply in natural Korean unless the user asks for another language.
2) Respect workspace isolation; never access paths outside the active workspace.
3) Do not invent credentials/secrets. Ask the user when required.
4) When users refer to you by any of your names or aliases, recognize that they are talking about you.
5) CRITICAL — NEVER answer factual questions from your own pretrained knowledge.
   You MUST use tools (web_search, search_memory, etc.) to find information first.
   If tools return no results, say "검색 결과를 찾지 못했습니다" — do NOT guess or fabricate.
   Your pretrained knowledge is outdated and unreliable. Only tool results are trustworthy.
   Exception: simple greetings, math, language translation, and creative writing do not require tools.
6) When a matching skill tool is available, call it DIRECTLY. Do not explain manually or web-search.

Your current capabilities:
{cap_block}
"""

    def build_task_prompt(self, task_text: str, prompt_class: str, tool_names: list[str]) -> str:
        base = ["Task execution contract:", f"- task: {task_text}", f"- class: {prompt_class}"]
        if tool_names:
            base.append(f"- allowed_tools: {', '.join(tool_names)}")
        else:
            base.append("- allowed_tools: (none)")
        class_rules: dict[str, list[str]] = {
            "direct_dialog": [
                "- reply naturally in Korean, short and direct",
                "- do not call tools",
            ],
            "research_focused": [
                "- you MUST call web_search FIRST before answering. NEVER skip search.",
                "- base your answer ONLY on search results, NOT your pretrained knowledge",
                "- if search returns empty or irrelevant results, say so honestly",
                "- include concise evidence and sources from search results",
                "- focus ONLY on the topic asked; do NOT introduce unrelated topics from memory",
                "- if the user asks to SAVE/저장 results to a file, you MUST call write_file IMMEDIATELY after search. Do NOT skip this step.",
                "- if the user asks to SCHEDULE/매일/매주 recurring tasks, you MUST call schedule_task with UTC time. Do NOT skip this step.",
            ],
            "factcheck_focused": [
                "- validate claims with explicit evidence from search results",
                "- separate verified facts from assumptions",
                "- if unverifiable, explain why briefly",
                "- focus ONLY on the topic asked; do NOT introduce unrelated topics from memory",
            ],
            "memory_focused": [
                "- CRITICAL: 'memory' means the workspace memory system (search_memory tool), NOT your pretrained LLM knowledge",
                "- RECALL intent ('내 이름이 뭐였지', '기억나', 'do you remember'): call search_memory to retrieve facts. Do NOT answer from pretrained knowledge.",
                "- STORE intent ('기억해', '기억해줘', '잘 기억', 'remember this'): call remember_fact(fact=...) IMMEDIATELY to save the user-provided fact. search_memory is optional (only to check duplicates). Do NOT skip remember_fact.",
                "- If search_memory returns empty results during recall, say '기억된 내용이 없습니다' or equivalent. Do NOT fabricate or guess",
                "- Only report what is actually stored in the memory system",
            ],
            "workspace_ops": [
                "- perform minimal safe workspace actions",
                "- avoid unrelated file changes",
                "- '워크스페이스 루트' 또는 '현재 폴더/파일 목록' 요청은 list_files를 호출해 실제 목록을 반환. 자의적인 설명만 주지 말 것.",
            ],
            "coding_ops": [
                "- search for up-to-date best practices/APIs first, then write code based on findings",
                "- you MUST produce actual working code in the final response, not just a plan or explanation",
                "- prefer precise edits over broad rewrites",
            ],
            "schedule_ops": [
                "- Distinguish three intents: REGISTER, LIST, CANCEL.",
                "- REGISTER ('매일/매주 X시에 ... 해줘'): call schedule_task EXACTLY ONCE. Do NOT search the web. Do NOT ask clarifying questions.",
                "  - Time conversion: KST = UTC+9. e.g., KST 08:00 = UTC 23:00 (previous day).",
                "  - schedule_task(run_at_utc='YYYY-MM-DDTHH:MM:SSZ', task_prompt='user request', recurrence='daily|weekly|monthly|none')",
                "  - task_prompt = user's original request, so the agent knows what to do when triggered.",
                "  - CRITICAL: call it ONCE. If you see a 'duplicate_in_turn' response, that means YOUR PREVIOUS CALL already registered the task just now — do NOT interpret this as 'the task was already scheduled before'. Confirm the fresh registration to the user.",
                "- LIST ('등록된 작업 뭐 있어?', '내 스케줄 보여줘', '예약된 거 알려줘', \"what's scheduled?\"): call list_scheduled_tasks. NEVER fabricate tasks. If empty, say so honestly.",
                "- CANCEL ('X 작업 취소해줘', '방금 등록한 ... 취소'): call list_scheduled_tasks FIRST to resolve the task_id, then IMMEDIATELY call cancel_scheduled_task(task_id=...). You MUST end with cancel_scheduled_task — listing alone is not enough. Do NOT call web_search for a cancel request.",
                "- After REGISTER, confirm what was scheduled, when (KST), and what will run.",
                "- After LIST, present the actual returned tasks ONLY — do NOT add fake examples like 'Database Backup' or 'Log Cleanup'.",
                "- After CANCEL, confirm which task_id was removed.",
                "- do NOT execute the scheduled action right now (no web_search, no actual work) — just manage the schedule.",
            ],
            "skill_ops": [
                "- CRITICAL: If a matching skill tool is available (e.g. srt_reserve), call it DIRECTLY with proper arguments.",
                "  Do NOT search the web. Do NOT explain how to do it manually. JUST CALL THE TOOL.",
                "- The skill tools are already installed and ready to use. Check allowed_tools list above.",
                "- For SRT: use srt_reserve tool with action='search' first, then action='reserve' with the train_index.",
                "  Read srt_id and srt_pw from config.json connectors section — they are already set up.",
                "- If no matching skill exists → search_hub(query) to find one on ClawHub",
                "  If found → install_skill(repo_url, skill_name) to download and auto-setup",
                "  If NOT found → tell the user: '해당 기능의 스킬이 ClawHub에 아직 등록되어 있지 않습니다.'",
                "- IMPORTANT: do NOT generate code or create skills yourself. You cannot make new skills.",
                "  External skills are installed ONLY via ClawHub. If ClawHub has no matching skill, the request cannot be fulfilled.",
                "- read credentials from config.json 'connectors' section, NEVER hardcode secrets in code",
            ],
            "config_ops": [
                "- MUST call update_config tool to save the user's settings. Do NOT just explain — actually save it.",
                "- Common config mappings:",
                "  brave api key → update_config(section='search', key='brave_api_key', value=...)",
                "  llm api key → update_config(section='llm', key='api_key', value=...)",
                "  telegram token → update_config(section='telegram', key='token', value=...)",
                "- Extract the key/value from the user's message. If they provide 'BSAxxxxx', that IS the value.",
                "- After saving, confirm what was saved (section.key) but never expose the full secret value.",
            ],
            "analysis_general": [
                "- for factual/informational questions: MUST call web_search first, answer based on results only",
                "- for creative tasks (stories, reports, plans): gather inspiration via search, then PRODUCE the requested deliverable in full",
                "- never answer factual questions from pretrained knowledge alone",
                "- if search returns no useful results, say '관련 정보를 찾지 못했습니다' honestly",
            ],
            "local_time": [
                "- provide current local system time directly",
                "- do not use web search for local time query",
            ],
        }
        base.extend(class_rules.get(prompt_class, class_rules["analysis_general"]))
        return "\n".join(base)

    def build_messages(self, user_text: str, source: str, user_id: str,
                       constraints: list[str] | None = None) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.build_system_prompt()}]
        layered = self.memory.recall_layers(user_text, per_layer=3)

        # Identity context
        identity_lines = [f"profile: agent_name={self.config.agent_name}"]
        if self.config.aliases:
            identity_lines.append(f"aliases: {', '.join(self.config.aliases)}")
        for row in layered["L1"]:
            c = str(row.get("content", "")).strip()
            if c:
                identity_lines.append(f"- {c}")

        # Domain context (query-relevant only, with scores)
        domain_lines: list[str] = []
        for layer in ("L2", "L3"):
            for row in layered[layer]:
                c = str(row.get("content", "")).strip()
                score = row.get("meta", {}).get("score", 0)
                if c and score > 0:
                    domain_lines.append(f"- [{layer}] {c}")

        context_parts = identity_lines[:]
        if domain_lines:
            context_parts.append("query_relevant_memory:")
            context_parts.extend(domain_lines[:6])
        if constraints:
            context_parts.append("job_constraints:")
            context_parts.extend([f"- {x}" for x in constraints if str(x).strip()])
        messages.append({"role": "system", "content": "Execution context:\n" + "\n".join(context_parts)})

        include_recent_history = not (source == "tester" and user_id == "eval")
        if include_recent_history:
            for row in self.memory.recent(self.config.max_recent_messages, source=source, user_id=user_id):
                role = row.get("role", "user")
                content = str(row.get("content", ""))
                if role in {"user", "assistant"} and content:
                    if role == "user" and content == user_text:
                        continue
                    if len(content) > 1800:
                        content = content[:900] + "\n...[history message compressed]...\n" + content[-400:]
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_text})
        return messages

    @staticmethod
    def replan_guard_message() -> dict[str, str]:
        return {
            "role": "system",
            "content": (
                "REPLAN CHECK:\n"
                "1) 방금 수집한 도구 결과로 사용자 요청을 충분히 해결할 수 있으면 지금 최종 답변을 작성하세요.\n"
                "2) 부족하다면 다음에 필요한 도구 1개만 호출하세요.\n"
                "3) web_search 결과의 중복도와 정보 충분성을 먼저 평가한 뒤, 부족할 때만 검색 축을 바꿔 재검색하세요.\n"
                "4) 불확실하면 불확실한 이유를 명시하고, 추가로 필요한 맥락을 1개만 요청하세요."
            ),
        }

    @staticmethod
    def append_replan_guard_once(messages: list[dict[str, Any]]) -> None:
        guard = PromptBuilder.replan_guard_message()
        target = str(guard.get("content") or "")
        for msg in reversed(messages):
            if str(msg.get("role")) != "system":
                continue
            content = str(msg.get("content") or "")
            if content == target:
                return
            if content.startswith("REPLAN CHECK:"):
                msg["content"] = target
                return
            break
        messages.append(guard)

    @staticmethod
    def prune_ephemeral_system_messages(messages: list[dict[str, Any]]) -> None:
        prefixes = (
            "Task execution contract:",
            "REPLAN CHECK:",
            "LOOP GUARDRAIL:",
            "수집된 정보가 답변 작성에 충분합니다.",
            "검색 결과가 충분히 수집되었습니다.",
            "검색 결과가 아직 불충분하거나 중복이 높습니다.",
        )
        kept: list[dict[str, Any]] = []
        for msg in messages:
            if str(msg.get("role")) != "system":
                kept.append(msg)
                continue
            content = str(msg.get("content") or "")
            if any(content.startswith(p) for p in prefixes):
                continue
            kept.append(msg)
        messages[:] = kept


# ═══════════════════════════════════════════════════════════════════════════════
# Section 10: ToolDispatcher
# ═══════════════════════════════════════════════════════════════════════════════

class ToolDispatcher:
    """Data-driven tool dispatch. Maps tool names to handler callables via a registry dict."""

    def __init__(self, tools: WorkspaceTools, log_fn: Callable, config: RuntimeConfig, client_ref: Any = None):
        self.tools = tools
        self._log = log_fn
        self.config = config
        self._client_ref = client_ref
        self._registry: dict[str, Callable] = self._build_registry()

    def _build_registry(self) -> dict[str, Callable]:
        t = self.tools
        return {
            "update_config": self._handle_update_config,
            "search_hub": lambda args: t.search_hub(query=str(args.get("query") or "")),
            "web_search": self._handle_web_search,
            "schedule_task": lambda args: t.schedule_task(
                run_at_utc=str(args.get("run_at_utc") or ""),
                task_prompt=str(args.get("task_prompt") or ""),
                recurrence=str(args.get("recurrence") or "auto"),
                interval=int(str(args.get("interval") or "1")) if str(args.get("interval") or "1").strip().lstrip("-").isdigit() else 1,
            ),
            "list_scheduled_tasks": lambda args: t.list_scheduled_tasks(
                include_done=bool(args.get("include_done", False)),
            ),
            "cancel_scheduled_task": lambda args: t.cancel_scheduled_task(
                task_id=str(args.get("task_id") or ""),
            ),
            "list_files": lambda args: t.list_files(path=str(args.get("path") or ".")),
            "read_file": lambda args: t.read_file(path=str(args.get("path") or "")),
            "write_file": lambda args: t.write_file(path=str(args.get("path") or ""), content=str(args.get("content") or "")),
            "search_memory": lambda args: t.search_memory(query=str(args.get("query") or ""), limit=int(args.get("limit") or 5)),
            "run_shell": lambda args: t.run_shell(command=str(args.get("command") or ""), timeout_sec=int(args.get("timeout_sec") or 30)),
            "install_skill": lambda args: t.install_skill(repo_url=str(args.get("repo_url") or ""), skill_name=args.get("skill_name")),
            "run_skill": lambda args: t.run_skill(skill_name=str(args.get("skill_name") or ""), args=str(args.get("args") or "")),
            "remember_fact": lambda args: t.remember_fact(
                fact=str(args.get("fact") or ""), layer=str(args.get("layer") or "L3"),
                relation=str(args.get("relation") or "related_to"), entity=str(args.get("entity") or "user"),
                target_entity=str(args.get("target_entity") or "")),
            "check_email": lambda args: t.check_email(
                provider=str(args.get("provider") or "gmail"), limit=int(args.get("limit") or 5),
                unread_only=bool(args.get("unread_only", True))),
            "check_calendar": lambda args: t.check_calendar(
                days_ahead=int(args.get("days_ahead") or 1), max_results=int(args.get("max_results") or 10)),
            "reset_workspace": lambda args: t.reset_workspace(confirm=str(args.get("confirm") or "")),
        }

    def _handle_update_config(self, args: dict[str, Any]) -> dict[str, Any]:
        result = self.tools.update_config(str(args.get("section")), str(args.get("key")), str(args.get("value")))
        if args.get("section") == "llm" and args.get("key") == "api_key":
            self.config.api_key = str(args.get("value"))
            if self._client_ref:
                self._client_ref.api_key = str(args.get("value"))
        return result

    def _handle_web_search(self, args: dict[str, Any]) -> dict[str, Any]:
        original_query = str(args.get("query") or "")
        search_type = str(args.get("search_type") or "auto")
        if IntentClassifier.is_local_time_query(original_query):
            return {"ok": False, "error": "local_time_query_should_not_use_web_search"}
        normalized_query = self.normalize_web_search_query(original_query)
        if normalized_query != original_query:
            self._log({"type": "web_search_query_rewrite", "original_query": original_query, "normalized_query": normalized_query})
        if not normalized_query:
            return {"ok": False, "error": "empty_query_after_normalization"}
        return self.tools.web_search(query=normalized_query, search_type=search_type)

    def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        tool_origin = "internal" if is_internal_tool_name(name) else "external"
        self._log({"type": "tool_execution_start", "tool_name": name, "tool_origin": tool_origin, "args": args})
        try:
            handler = self._registry.get(name)
            if handler:
                result = handler(args)
            elif not is_internal_tool_name(name):
                result = self.tools.run_dynamic_tool(name, args)
            else:
                result = {"ok": False, "error": f"unknown tool: {name}"}
        except Exception as e:
            result = {"ok": False, "error": f"{type(e).__name__}: {safe_str(e)}"}
            self._log({"type": "tool_execution_end", "tool_name": name, "tool_origin": tool_origin, "ok": False, "error": result.get("error", "")})
            return result
        self._log({
            "type": "tool_execution_end", "tool_name": name, "tool_origin": tool_origin,
            "ok": bool(result.get("ok")) if isinstance(result, dict) else False,
            "error": str(result.get("error") or "") if isinstance(result, dict) else "",
        })
        return result

    @staticmethod
    def normalize_web_search_query(query: str) -> str:
        q = (query or "").strip()
        if not q:
            return ""
        if IntentClassifier.is_local_time_query(q):
            return ""
        low = q.lower()
        # Strip HTML-like decorations that may leak from previous results
        q = re.sub(r"</?strong>", "", q)
        # Korean news/trend queries: append context keywords
        if any(k in q for k in ["동향", "현황", "전망", "근황", "최신"]) and ("뉴스" not in q):
            q += " 뉴스"
        if ("주식" in q or "증시" in q) and not any(k in q for k in ["주가", "시세", "뉴스", "동향"]):
            q += " 주가 동향"
        if any(k in low for k in ["paper", "research", "논문", "연구"]) and ("latest" not in low and "최신" not in q):
            q += " 최신"
        return q[:400]

    @staticmethod
    def search_result_stats(result: dict[str, Any]) -> tuple[int, int]:
        hits = result.get("results", []) if isinstance(result, dict) else []
        if not isinstance(hits, list):
            return 0, 0
        unique_keys: set[str] = set()
        for h in hits:
            if not isinstance(h, dict):
                continue
            url = str(h.get("url") or "").strip()
            title = str(h.get("title") or "").strip().lower()
            key = url or title
            if key:
                unique_keys.add(key)
        return len(hits), len(unique_keys)

    @staticmethod
    def compact_tool_result(name: str, result: dict[str, Any], max_items: int = 3) -> dict[str, Any]:
        if not isinstance(result, dict):
            return {"ok": False, "error": "invalid_tool_result"}
        if name == "search_memory":
            hits = result.get("hits", [])
            compact_hits: list[dict[str, Any]] = []
            if isinstance(hits, list):
                for row in hits[:max_items]:
                    if not isinstance(row, dict):
                        continue
                    compact_hits.append({"ts": str(row.get("ts") or ""), "role": str(row.get("role") or ""), "content": safe_str(str(row.get("content") or ""))[:220]})
            return {"ok": bool(result.get("ok")), "error": str(result.get("error") or ""), "hits": compact_hits}
        if name == "web_search":
            rows = result.get("results", [])
            compact_rows: list[dict[str, Any]] = []
            if isinstance(rows, list):
                for row in rows[:max_items]:
                    if not isinstance(row, dict):
                        continue
                    entry: dict[str, Any] = {
                        "title": safe_str(str(row.get("title") or ""))[:160],
                        "url": safe_str(str(row.get("url") or ""))[:220],
                        "snippet": safe_str(str(row.get("snippet") or ""))[:300],
                    }
                    if row.get("age"):
                        entry["age"] = str(row["age"])
                    if row.get("source"):
                        entry["source"] = str(row["source"])[:60]
                    extras = row.get("extra_snippets", [])
                    if extras and isinstance(extras, list):
                        entry["extra_snippets"] = [safe_str(s)[:200] for s in extras[:3]]
                    compact_rows.append(entry)
            return {"ok": bool(result.get("ok")), "source": str(result.get("source") or ""), "search_type": str(result.get("search_type") or ""), "error": str(result.get("error") or ""), "results": compact_rows}
        # For external skills (run_skill/run_dynamic_tool), parse stdout JSON
        if "stdout" in result and isinstance(result.get("stdout"), str):
            try:
                parsed = json.loads(result["stdout"])
                if isinstance(parsed, dict):
                    # Compact batch results: keep summary + limit list items
                    for k, v in parsed.items():
                        if isinstance(v, list) and len(v) > 5:
                            parsed[k] = v[:5] + [f"... and {len(v) - 5} more"]
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass

        compact: dict[str, Any] = {}
        for k, v in result.items():
            if isinstance(v, str):
                compact[k] = safe_str(v)[:400]
            elif isinstance(v, list):
                compact[k] = v[:max_items]
            elif isinstance(v, dict):
                compact[k] = {kk: (safe_str(vv)[:200] if isinstance(vv, str) else vv) for kk, vv in list(v.items())[:10]}
            else:
                compact[k] = v
        return compact


# ═══════════════════════════════════════════════════════════════════════════════
# Section 11: JobBreakerEngine (uses IntentClassifier, no duplicated methods)
# ═══════════════════════════════════════════════════════════════════════════════

class JobBreakerEngine:
    def __init__(self, client: LLMClient):
        self.client = client

    def _normalize_plan(self, user_text: str, plan: dict[str, Any]) -> dict[str, Any]:
        refined_goal = str(plan.get("refined_goal") or user_text).strip() or user_text
        raw_constraints = plan.get("constraints")
        constraints: list[str] = []
        if isinstance(raw_constraints, list):
            deny_patterns = [
                r"\b(no|cannot|can't|unable|불가|없다|못)\b.*\b(real[- ]?time|시간)\b",
                r"\b(i|assistant|ai)\b.*\b(cannot|can't|unable)\b",
                r"\b실시간\b.*\b불가|불가능\b",
            ]
            for c in raw_constraints:
                text = str(c).strip()
                if not text:
                    continue
                low = text.lower()
                if any(re.search(p, low) for p in deny_patterns):
                    continue
                constraints.append(text)

        raw_tasks = plan.get("tasks")
        tasks: list[str] = []
        if isinstance(raw_tasks, list):
            deny_task_patterns = [
                r"\b(cannot|can't|unable)\b.*\b(provide|answer|time)\b",
                r"\b알려줄 수 없|제공할 수 없|불가\b",
                r"\b(대신|alternative)\b.*\b(check|확인)\b",
            ]
            for t in raw_tasks:
                text = str(t).strip()
                if not text:
                    continue
                low = text.lower()
                if any(re.search(p, low) for p in deny_task_patterns):
                    continue
                tasks.append(text)

        if IntentClassifier.is_local_time_query(user_text):
            if not any(("time" in t.lower()) or ("시간" in t) for t in tasks):
                tasks = ["Check local system time first and report it in Korean."]
        elif IntentClassifier.is_memory_recall_query(user_text):
            tasks = [
                "Use the search_memory tool to retrieve relevant facts from workspace memory. "
                "Answer ONLY based on what search_memory returns. "
                "Do NOT use your pretrained knowledge to answer memory questions. "
                "If search_memory returns no results, clearly state that nothing is stored in memory."
            ]
        elif IntentClassifier.is_memory_store_query(user_text):
            tasks = ["Remember the user-provided fact accurately and confirm it in Korean."]
        elif IntentClassifier.should_force_web_grounding(user_text):
            if not any(("search" in t.lower()) or ("검색" in t) or ("출처" in t) or ("근거" in t) for t in tasks):
                tasks.insert(0, "Collect up-to-date evidence with web search before drafting the final answer.")

        if not tasks:
            tasks = [user_text]
        tasks = self._dedup_tasks(tasks)
        return {"refined_goal": refined_goal, "constraints": constraints[:8], "tasks": tasks[:6]}

    @staticmethod
    def _dedup_tasks(tasks: list[str]) -> list[str]:
        if len(tasks) <= 1:
            return tasks
        _STOP = {
            "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "with",
            "is", "are", "was", "were", "be", "been", "being", "it", "its", "this",
            "that", "from", "by", "at", "as", "all", "about", "into", "through",
            "search", "find", "check", "look", "get", "collect", "gather", "compile",
            "summarize", "provide", "report", "analyze", "update", "latest", "recent",
            "current", "new", "using", "based", "before", "after", "then", "also",
            "검색", "조사", "수집", "정리", "요약", "분석", "확인", "파악", "조회",
            "최신", "최근", "현재", "관련", "대한", "통해", "위해", "해서", "하여",
        }
        _SYNONYMS: list[set[str]] = [
            {"war", "conflict", "battle", "combat", "전쟁", "분쟁", "충돌", "교전"},
            {"news", "updates", "developments", "situation", "소식", "동향", "상황", "현황", "근황", "전개"},
            {"military", "armed", "forces", "army", "군사", "군", "무장"},
            {"economy", "economic", "market", "경제", "시장", "증시"},
            {"summary", "overview", "comprehensive", "종합", "개요", "전체"},
        ]
        _SYN_MAP: dict[str, str] = {}
        for group in _SYNONYMS:
            canonical = min(group)
            for w in group:
                _SYN_MAP[w] = canonical

        def _content_words(text: str) -> set[str]:
            raw = text.lower().replace("-", " ").replace("_", " ")
            tokens = {t.strip(".,;:!?()[]{}\"'") for t in raw.split()}
            normalized = set()
            for t in tokens:
                if len(t) < 2 or t in _STOP:
                    continue
                normalized.add(_SYN_MAP.get(t, t))
            return normalized

        kept: list[str] = [tasks[0]]
        kept_cw: list[set[str]] = [_content_words(tasks[0])]
        for task in tasks[1:]:
            t_cw = _content_words(task)
            is_dup = False
            for existing_cw in kept_cw:
                if not t_cw or not existing_cw:
                    continue
                intersection = t_cw & existing_cw
                jaccard = len(intersection) / len(t_cw | existing_cw)
                subset_ratio = len(intersection) / len(t_cw) if t_cw else 0
                if jaccard > 0.35 or subset_ratio > 0.6:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(task)
                kept_cw.append(t_cw)
        return kept

    def break_job(self, user_text: str, planning_context: str = "") -> dict[str, Any]:
        context_block = ""
        if planning_context.strip():
            context_block = (
                "\nPlanning Evidence (from pre-search/tool checks):\n"
                f"{planning_context.strip()}\n"
                "Use this evidence to make a more executable, grounded plan.\n"
            )
        prompt = (
            "Analyze the User Intent and break it down into a structured JSON execution plan.\n"
            "You must maximize feasibility: create a plan that can actually be executed with tools and grounded evidence.\n"
            "Do NOT solve from model memory when fresh/verified info may be needed.\n"
            "Prefer proactive evidence collection (web search, file/system checks) before final answering.\n"
            "If the request is simple (e.g. greeting, basic question), output one concise executable task.\n"
            "CRITICAL planning rules:\n"
            "1) Never put self-capability disclaimers in constraints/tasks (e.g., 'cannot provide current time').\n"
            "2) Constraints must be external/objective limits only (permissions, missing inputs, policy/safety).\n"
            "3) Tasks must be action-oriented and executable, not apology or refusal statements.\n"
            "4) For current/real-time/fresh-info queries, first task should gather up-to-date evidence.\n"
            "5) For environment-observable requests (e.g., local time, files, system state),"
            " prefer local system/tool checks over generic web lookup.\n"
            "6) NEVER create duplicate or overlapping tasks. Each task must have a clearly distinct purpose.\n"
            "   BAD: Task1='Search Iran war news', Task2='Search latest Iran conflict updates' (same topic)\n"
            "   GOOD: Task1='Search and compile latest Iran war developments' (single comprehensive task)\n"
            "7) Prefer fewer, broader tasks over many narrow ones. For a single-topic query, use 1 task.\n"
            "   Only split into multiple tasks when the user asks about genuinely different topics.\n"
            "8) Stay focused on the user's SPECIFIC question. Do not expand scope to tangentially related topics.\n"
            "   If user asks about 'US-Iran war', do NOT add tasks about Ukraine, North Korea, etc.\n"
            "Format exactly as JSON:\n"
            "{\n"
            '  "refined_goal": "...",\n'
            '  "constraints": ["..."],\n'
            '  "tasks": ["Task 1", "Task 2"]\n'
            "}\n"
            f"{context_block}"
            f"User Intent: {user_text}\n"
        )
        try:
            out = self.client.chat([
                {"role": "system", "content": "You are a planning module. Output raw JSON ONLY. No markdown blocks."},
                {"role": "user", "content": prompt}
            ], timeout_sec=float(LLM_QUERY_TIMEOUT_SEC))
            text = out.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                parsed = json.loads(text[start:end+1])
                if isinstance(parsed, dict):
                    return self._normalize_plan(user_text, parsed)
        except Exception as e:
            print(f"[JobBreaker] parse failed: {e}")
        return self._normalize_plan(user_text, {"refined_goal": user_text, "constraints": [], "tasks": [user_text]})


# ═══════════════════════════════════════════════════════════════════════════════
# Section 12: TaskExecutor (split into sub-methods)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExecutionContext:
    """Mutable state for a single task execution loop."""
    selected_tools: list[dict[str, Any]]
    selected_tool_names: list[str]
    prompt_class: str
    route_meta: dict[str, Any]
    should_force_search: bool = False
    should_force_memory: bool = False
    forced_search_query: str = ""
    last_tool_note: str = ""
    compression_events: int = 0
    total_tool_calls: int = 0
    tool_signature_counts: dict[str, int] = field(default_factory=dict)
    seen_search_urls: set[str] = field(default_factory=set)
    seen_search_titles: set[str] = field(default_factory=set)
    force_finalize_mode: bool = False
    finalize_notice_pushed: bool = False
    context_reask_streak: int = 0
    loop_guard_pushed: bool = False
    forced_search_done: bool = False
    forced_memory_done: bool = False
    memory_write_done: bool = False
    reflection_done: bool = False
    skepticism_done: bool = False
    associative_followup_done: bool = False
    skeptical_followup_done: bool = False
    remembered_facts: set[str] = field(default_factory=set)
    # Per-turn path-level write counter. write_file with the same target path
    # beyond this limit is rejected so a runaway rewrite loop cannot exhaust
    # the tool-iteration budget.
    write_file_path_counts: dict[str, int] = field(default_factory=dict)
    # Per-turn total count per tool name (name-level, args-independent). Used
    # to cut off oscillating loops where args differ but the intent is the
    # same tool being hammered (e.g., repeated web_search with tiny rewrites).
    tool_name_counts: dict[str, int] = field(default_factory=dict)


_WRITE_FILE_PATH_LIMIT = 5
_TOOL_NAME_LIMIT = 6


class TaskExecutor:
    # ── Response Reflection: expected tool calls per prompt_class ─────────────
    _REFLECTION_EXPECTED_TOOLS: dict[str, set[str]] = {
        "research_focused": {"web_search"},
        "factcheck_focused": {"web_search"},
        "memory_focused": {"search_memory"},
        "schedule_ops": {"schedule_task"},
        "email_ops": {"check_email"},
    }
    _REFLECTION_SKIP_CLASSES: set[str] = {"direct_dialog", "local_time", "config_ops"}
    _MODEL_FAMILY_TERMS = [
        "gpt", "o1", "o3", "o4", "claude", "gemini", "llama", "qwen",
        "gemma", "deepseek", "mistral", "grok", "command r", "phi",
    ]
    _SKEPTIC_PERSONAS: list[dict[str, str]] = [
        {
            "id": "coverage_auditor",
            "name": "Coverage Auditor",
            "focus": "Check missing entities/categories and incomplete scope coverage.",
        },
        {
            "id": "evidence_prosecutor",
            "name": "Evidence Prosecutor",
            "focus": "Challenge unsupported claims, weak grounding, and source mismatch.",
        },
        {
            "id": "bias_sentinel",
            "name": "Bias Sentinel",
            "focus": "Detect regional/vendor bias, one-sided framing, and imbalance.",
        },
    ]

    def __init__(self, engine: 'AgentEngine'):
        self.engine = engine
        self.config = engine.config
        self.client = engine.client
        self.dispatcher = engine.dispatcher
        self.context_mgr = engine.context_mgr
        self.prompt_builder = engine.prompt_builder

    @staticmethod
    def _is_model_landscape_query(query: str) -> bool:
        low = (query or "").lower()
        topic_hit = any(k in low for k in ["llm", "언어모델", "language model", "모델", "model"])
        enum_hit = any(k in low for k in ["최신", "latest", "비교", "목록", "리스트", "top", "추천"])
        return topic_hit and enum_hit

    @classmethod
    def _extract_model_family_hits(cls, text: str) -> set[str]:
        low = (text or "").lower()
        out: set[str] = set()
        for t in cls._MODEL_FAMILY_TERMS:
            if t in low:
                out.add(t)
        return out

    def _build_associative_query_from_memory(self, user_query: str, memory_result: dict[str, Any]) -> str:
        if not self._is_model_landscape_query(user_query):
            return ""
        hits = memory_result.get("hits", []) if isinstance(memory_result, dict) else []
        if not isinstance(hits, list):
            return ""
        base_families = self._extract_model_family_hits(user_query)
        mem_families: set[str] = set()
        for h in hits[:8]:
            if not isinstance(h, dict):
                continue
            mem_families.update(self._extract_model_family_hits(str(h.get("content") or "")))
        extra = [f for f in sorted(mem_families) if f not in base_families]
        if not extra:
            return ""
        is_ko = bool(re.search(r"[\uac00-\ud7a3]", user_query or ""))
        if is_ko:
            return f"{user_query} {' '.join(extra[:4])} 포함 비교"
        return f"{user_query} include {' '.join(extra[:4])} comparison"

    def _build_skeptical_followup_query(self, user_query: str, search_result: dict[str, Any]) -> tuple[str, str]:
        hits = search_result.get("results", []) if isinstance(search_result, dict) else []
        if not isinstance(hits, list) or not hits:
            return "", ""
        text_blob = " ".join(f"{h.get('title', '')} {h.get('snippet', '')}" for h in hits if isinstance(h, dict))
        families = self._extract_model_family_hits(text_blob)
        is_ko = bool(re.search(r"[\uac00-\ud7a3]", user_query or ""))

        if self._is_model_landscape_query(user_query) and len(families) < 4:
            q = "latest LLM models GPT Claude Gemini Llama Qwen Gemma DeepSeek Mistral comparison"
            if is_ko:
                q = "최신 LLM 모델 GPT Claude Gemini Llama Qwen Gemma DeepSeek Mistral 비교"
            return q, "model_family_coverage_gap"

        tlds: set[str] = set()
        for h in hits:
            if not isinstance(h, dict):
                continue
            url = str(h.get("url") or "").strip().lower()
            m = re.search(r"https?://([^/]+)", url)
            if not m:
                continue
            host = m.group(1)
            parts = host.split(".")
            if parts:
                tlds.add(parts[-1])
        if len(tlds) <= 1 and len(hits) >= 3:
            q = f"{user_query} global sources"
            if is_ko:
                q = f"{user_query} 글로벌 출처 비교"
            return q, "source_diversity_gap"
        return "", ""

    def _ingest_search_hits(self, ctx: ExecutionContext, result: dict[str, Any]) -> tuple[int, int]:
        hits = result.get("results", []) if isinstance(result, dict) else []
        if not isinstance(hits, list):
            return 0, 0
        total_hits = 0
        new_hits = 0
        for h in hits:
            if not isinstance(h, dict):
                continue
            total_hits += 1
            url = str(h.get("url") or "").strip()
            title = str(h.get("title") or "").strip().lower()
            is_new = False
            if url and url not in ctx.seen_search_urls:
                ctx.seen_search_urls.add(url)
                is_new = True
            if title and title not in ctx.seen_search_titles:
                ctx.seen_search_titles.add(title)
                is_new = True
            if is_new:
                new_hits += 1
        return total_hits, new_hits

    def _init_context(self, messages: list[dict[str, Any]], user_original_text: str) -> ExecutionContext:
        self.prompt_builder.prune_ephemeral_system_messages(messages)
        search_query = self.engine._latest_user_query(messages)
        forced_search_query = ToolDispatcher.normalize_web_search_query(search_query)
        selected_tools, selected_tool_names, prompt_class, route_meta = self.engine._select_tools_for_query(search_query)

        # If the original (pre-decomposition) query implies a "sticky" intent, ensure those
        # tools and that prompt class are preserved across all sub-tasks. Without this, a
        # request like "매일 8시 뉴스 검색해서 요약 브리핑" gets decomposed by JobBreaker into
        # "검색" / "요약" / "브리핑" sub-tasks, each of which routes to research_focused
        # and loses access to schedule_task — so the agent never registers the recurring task.
        STICKY_INTENTS = ("skill_ops", "coding_ops", "schedule_ops")
        if user_original_text and user_original_text != search_query:
            _, orig_names, orig_class, orig_meta = self.engine._select_tools_for_query(user_original_text)
            if orig_class in STICKY_INTENTS:
                # Merge original tools into sub-task tools
                for oname in orig_names:
                    if oname not in selected_tool_names:
                        selected_tool_names.append(oname)
                for spec in self.engine.tools.get_all_tools():
                    fn_name = str(spec.get("function", {}).get("name", ""))
                    if fn_name in orig_names and spec not in selected_tools:
                        selected_tools.append(spec)
                if prompt_class not in STICKY_INTENTS:
                    prompt_class = orig_class
                route_meta["routing_override"] = f"original_intent({orig_class})_tools_merged"

        if user_original_text:
            all_names = [str(s.get("function", {}).get("name", "")) for s in self.engine.tools.get_all_tools() if str(s.get("function", {}).get("name", ""))]
            user_prohibited = IntentClassifier.parse_tool_prohibitions(user_original_text.lower(), all_names)
            if user_prohibited:
                selected_tools = [s for s in selected_tools if str(s.get("function", {}).get("name", "")) not in user_prohibited]
                selected_tool_names = [n for n in selected_tool_names if n not in user_prohibited]
                route_meta["user_prohibited_tools"] = list(user_prohibited)

        should_force_search = IntentClassifier.should_force_web_grounding(search_query) and prompt_class not in {
            "direct_dialog", "local_time", "memory_focused", "workspace_ops",
        }
        should_force_memory = (
            (prompt_class == "memory_focused")
            or IntentClassifier.is_memory_recall_query(search_query)
        )

        messages.append({"role": "system", "content": self.prompt_builder.build_task_prompt(search_query, prompt_class, selected_tool_names)})
        self.engine._log_runtime({"type": "execution_strategy", "query": search_query, "prompt_class": prompt_class, "allowed_tools": selected_tool_names, "route_meta": route_meta})

        return ExecutionContext(
            selected_tools=selected_tools, selected_tool_names=selected_tool_names,
            prompt_class=prompt_class, route_meta=route_meta,
            should_force_search=should_force_search, should_force_memory=should_force_memory,
            forced_search_query=forced_search_query,
        )

    def _handle_local_time_fastpath(self, source: str, user_id: str) -> str:
        now_local = datetime.now().astimezone()
        tz_name = now_local.tzname() or "LOCAL"
        hhmm = now_local.strftime("%H:%M")
        text = f"현재 로컬 시스템 시간({tz_name})은 {now_local.strftime('%Y-%m-%d %H:%M:%S')} 입니다. (HH:MM {hhmm})"
        self.engine._log_runtime({"type": "response_complete", "source": source, "user_id": user_id, "iterations": 0, "tool_calls": 0, "compression_events": 0, "mode": "local_time_fastpath"})
        return text

    def _manage_context(self, messages: list[dict[str, Any]], ctx: ExecutionContext, iter_idx: int) -> None:
        pre_compress_chars = self.context_mgr.messages_char_count(messages)
        compressed, notes = self.context_mgr.maybe_compress_context(messages)
        if compressed:
            ctx.compression_events += 1
            post_compress_chars = self.context_mgr.messages_char_count(messages)
            self.engine._log_runtime({"type": "context_compression", "iteration": iter_idx, "notes": notes, "pre_chars": pre_compress_chars, "post_chars": post_compress_chars})
        current_chars = self.context_mgr.messages_char_count(messages)
        if current_chars > self.config.context_max_chars:
            hard_notes = self.context_mgr.aggressive_context_cut(messages)
            if hard_notes:
                self.engine._log_runtime({"type": "context_hard_cut", "iteration": iter_idx, "notes": hard_notes, "chars": self.context_mgr.messages_char_count(messages)})
        orphan_removed = self.context_mgr.sanitize_orphan_tool_calls(messages)
        if orphan_removed > 0:
            self.engine._log_runtime({"type": "tool_call_sanitized", "iteration": iter_idx, "removed_count": orphan_removed})

    def _try_forced_bootstrap(self, messages: list[dict[str, Any]], ctx: ExecutionContext, iter_idx: int) -> bool:
        if ctx.total_tool_calls != 0 or iter_idx > 2:
            return False
        did_bootstrap = False
        if ctx.should_force_memory and not ctx.forced_memory_done:
            ctx.forced_memory_done = True
            search_query = self.engine._latest_user_query(messages)
            self.engine._log_runtime({
                "type": "tool_call_origin",
                "iteration": iter_idx,
                "internal_calls": 1,
                "external_calls": 0,
                "internal_tools": ["search_memory"],
                "external_tools": [],
                "tool_call_details": [{"name": "search_memory", "args": {"query": search_query[:200], "limit": 5}}],
            })
            mem_result = self.dispatcher.dispatch("search_memory", {"query": search_query[:200], "limit": 5})
            ctx.total_tool_calls += 1
            did_bootstrap = True
            self.engine._log_runtime({"type": "forced_memory_bootstrap", "iteration": iter_idx, "query": search_query[:200], "ok": bool(mem_result.get("ok")) if isinstance(mem_result, dict) else False})
            sanitized_mem = ToolDispatcher.compact_tool_result("search_memory", mem_result if isinstance(mem_result, dict) else {})
            messages.append({"role": "system", "content": "Tool result (search_memory):\n" + safe_json_dumps(sanitized_mem, ensure_ascii=False)[:2500]})
            ctx.last_tool_note = safe_json_dumps(sanitized_mem, ensure_ascii=False)[:800]
            if ctx.should_force_search and not ctx.associative_followup_done:
                assoc_q = self._build_associative_query_from_memory(search_query, mem_result if isinstance(mem_result, dict) else {})
                if assoc_q and assoc_q != ctx.forced_search_query:
                    self.engine._log_runtime({
                        "type": "tool_call_origin",
                        "iteration": iter_idx,
                        "internal_calls": 1,
                        "external_calls": 0,
                        "internal_tools": ["web_search"],
                        "external_tools": [],
                        "tool_call_details": [{"name": "web_search", "args": {"query": assoc_q}}],
                    })
                    assoc_result = self.dispatcher.dispatch("web_search", {"query": assoc_q})
                    ctx.total_tool_calls += 1
                    ctx.associative_followup_done = True
                    self.engine._log_runtime({
                        "type": "associative_search_followup",
                        "iteration": iter_idx,
                        "query": assoc_q[:220],
                        "ok": bool(assoc_result.get("ok")) if isinstance(assoc_result, dict) else False,
                    })
                    self._ingest_search_hits(ctx, assoc_result if isinstance(assoc_result, dict) else {})
                    sanitized_assoc = ToolDispatcher.compact_tool_result("web_search", assoc_result if isinstance(assoc_result, dict) else {})
                    messages.append({"role": "system", "content": "Tool result (web_search_associative):\n" + safe_json_dumps(sanitized_assoc, ensure_ascii=False)[:2500]})
                    ctx.last_tool_note = safe_json_dumps(sanitized_assoc, ensure_ascii=False)[:800]

        if ctx.should_force_search and ctx.forced_search_query and not ctx.forced_search_done:
            ctx.forced_search_done = True
            self.engine._log_runtime({
                "type": "tool_call_origin",
                "iteration": iter_idx,
                "internal_calls": 1,
                "external_calls": 0,
                "internal_tools": ["web_search"],
                "external_tools": [],
                "tool_call_details": [{"name": "web_search", "args": {"query": ctx.forced_search_query}}],
            })
            web_result = self.dispatcher.dispatch("web_search", {"query": ctx.forced_search_query})
            ctx.total_tool_calls += 1
            did_bootstrap = True
            self.engine._log_runtime({"type": "forced_search_bootstrap", "iteration": iter_idx, "query": ctx.forced_search_query[:200], "ok": bool(web_result.get("ok")) if isinstance(web_result, dict) else False})
            self._ingest_search_hits(ctx, web_result if isinstance(web_result, dict) else {})
            sanitized_web = ToolDispatcher.compact_tool_result("web_search", web_result if isinstance(web_result, dict) else {})
            messages.append({"role": "system", "content": "Tool result (web_search):\n" + safe_json_dumps(sanitized_web, ensure_ascii=False)[:2500]})
            ctx.last_tool_note = safe_json_dumps(sanitized_web, ensure_ascii=False)[:800]

        if did_bootstrap:
            PromptBuilder.append_replan_guard_once(messages)
        return did_bootstrap

    def _handle_tool_calls(self, messages: list[dict[str, Any]], ctx: ExecutionContext, tool_calls: list[dict[str, Any]], text: str, iter_idx: int) -> None:
        """Execute tool calls and update context. Modifies messages in-place."""
        ctx.total_tool_calls += len(tool_calls)
        internal_calls, external_calls = split_tool_calls_by_origin(tool_calls)
        # Build a parallel list of (name, args) so IPC consumers can validate args, not just names
        tool_call_details: list[dict[str, Any]] = []
        for tc in tool_calls:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            name = str(fn.get("name") or "")
            raw_args = fn.get("arguments", "{}")
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except Exception:
                parsed_args = {}
            if not isinstance(parsed_args, dict):
                parsed_args = {}
            tool_call_details.append({"name": name, "args": parsed_args})
        self.engine._log_runtime({
            "type": "tool_call_origin", "iteration": iter_idx,
            "internal_calls": len(internal_calls), "external_calls": len(external_calls),
            "internal_tools": [str(tc.get("function", {}).get("name") or "") for tc in internal_calls],
            "external_tools": [str(tc.get("function", {}).get("name") or "") for tc in external_calls],
            "tool_call_details": tool_call_details,
        })

        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception:
                args = {}
            if not isinstance(args, dict):
                args = {}
            signature = ToolCallParser.tool_signature(name, args if isinstance(args, dict) else {})
            ctx.tool_signature_counts[signature] = ctx.tool_signature_counts.get(signature, 0) + 1
            ctx.tool_name_counts[name] = ctx.tool_name_counts.get(name, 0) + 1

            # Path-level dedupe for write_file: args-based signature dedup
            # lets tiny content edits bypass the limit, so we cap rewrites per
            # target path as well.
            write_path = ""
            if name == "write_file":
                write_path = str(args.get("path", "")).strip()
                if write_path:
                    prior = ctx.write_file_path_counts.get(write_path, 0)
                    ctx.write_file_path_counts[write_path] = prior + 1

            if ctx.selected_tool_names and name not in ctx.selected_tool_names and is_internal_tool_name(name):
                result = {"ok": False, "error": f"Tool '{name}' is not in the allowed tool set for this task. Use only: {', '.join(ctx.selected_tool_names)}"}
            elif name == "write_file" and write_path and ctx.write_file_path_counts[write_path] > _WRITE_FILE_PATH_LIMIT:
                # The earlier successful writes already landed on disk; this
                # cap only blocks further rewrites in the same turn. Make
                # that explicit so the model doesn't relay 'file save is
                # blocked by the system' to the user.
                result = {
                    "ok": True,
                    "dedup": True,
                    "write_cap_reached": True,
                    "path": write_path,
                    "note": (
                        f"You already wrote to '{write_path}' {_WRITE_FILE_PATH_LIMIT} times in THIS turn. "
                        "The earlier write(s) are already saved to disk. Stop rewriting and "
                        "tell the user the file is saved; do NOT claim the system blocks saving."
                    ),
                }
                ctx.force_finalize_mode = True
            elif ctx.tool_name_counts.get(name, 0) > _TOOL_NAME_LIMIT and is_internal_tool_name(name):
                result = {"ok": False, "error": f"도구 '{name}'를 이미 {_TOOL_NAME_LIMIT}회 호출했습니다. 결과를 정리해 사용자에게 답변하세요."}
                ctx.force_finalize_mode = True
            elif name == "remember_fact":
                fact_key = str(args.get("fact", "")).strip().lower()
                if fact_key and fact_key in ctx.remembered_facts:
                    result = {"ok": True, "dedup": True, "fact": args.get("fact", ""), "note": "already_remembered_in_this_turn"}
                    ctx.force_finalize_mode = True
                else:
                    result = self.dispatcher.dispatch(name, args)
                    if isinstance(result, dict) and result.get("ok") and fact_key:
                        ctx.remembered_facts.add(fact_key)
                        if ctx.prompt_class == "memory_focused":
                            ctx.memory_write_done = True
                            ctx.force_finalize_mode = True
            elif ctx.tool_signature_counts[signature] > 1:
                # Per-turn duplicate: the FIRST call's result is still valid.
                # We return ok=True with an explicit note so the model does
                # not mistake this for 'the task is already registered' and
                # relay a confusing 'already scheduled' message to the user.
                result = {
                    "ok": True,
                    "dedup": True,
                    "duplicate_in_turn": True,
                    "note": (
                        f"'{name}' was already called with identical args earlier in THIS turn. "
                        "The earlier call's side effects (e.g., schedule insertion, fact save) are "
                        "already applied. Do NOT tell the user the task 'was already scheduled from "
                        "before' — it was scheduled just now by your previous call. Reuse the prior "
                        "result and write the final confirmation message to the user."
                    ),
                }
                ctx.force_finalize_mode = True
            else:
                result = self.dispatcher.dispatch(name, args)

            if name == "web_search":
                self._process_search_result(messages, ctx, result, iter_idx)

            sanitized_result = ToolDispatcher.compact_tool_result(name, result)
            ctx.last_tool_note = safe_json_dumps(sanitized_result, ensure_ascii=False)[:800]
            messages.append({"role": "system", "content": f"Tool result ({name}):\n" + safe_json_dumps(sanitized_result, ensure_ascii=False)[:2500]})
            if name == "remember_fact" and ctx.memory_write_done:
                messages.append({"role": "system", "content": "memory_focused 저장이 완료되었습니다. 추가 도구 호출 없이 저장 결과를 한 줄로 확인하고 최종 답변을 종료하세요."})

        # Check for context reask in assistant text alongside tool calls
        if IntentClassifier.is_context_reask_text(text):
            ctx.context_reask_streak += 1
            late_loop_phase = iter_idx >= max(3, self.config.max_tool_iterations // 2 + 1)
            if late_loop_phase and ctx.context_reask_streak >= 3 and len(ctx.seen_search_urls) > 0 and ctx.total_tool_calls >= 4:
                ctx.force_finalize_mode = True
                self.engine._log_runtime({"type": "loop_guardrail", "iteration": iter_idx, "reason": "context_reask_detected_in_assistant_output", "tool_calls": ctx.total_tool_calls, "seen_search_urls": len(ctx.seen_search_urls)})
                if not ctx.loop_guard_pushed:
                    messages.append({"role": "system", "content": "LOOP GUARDRAIL:\n동일한 맥락 재질문이 반복되고 있습니다. 지금까지 수집한 결과만으로 최종 답변을 작성하세요. 추가 도구 호출 없이 답변을 종료하세요."})
                    ctx.loop_guard_pushed = True
        else:
            ctx.context_reask_streak = 0
        PromptBuilder.append_replan_guard_once(messages)

    def _process_search_result(self, messages: list[dict[str, Any]], ctx: ExecutionContext, result: dict[str, Any], iter_idx: int) -> None:
        hits = result.get("results", []) if isinstance(result, dict) else []
        if not isinstance(hits, list) or not hits:
            return
        total_hits, new_hits = self._ingest_search_hits(ctx, result if isinstance(result, dict) else {})
        hit_count, uniq_count = ToolDispatcher.search_result_stats(result if isinstance(result, dict) else {})
        last_user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_text = str(m.get("content", ""))
                break
        if not ctx.skeptical_followup_done:
            skeptical_q, skeptical_reason = self._build_skeptical_followup_query(last_user_text, result if isinstance(result, dict) else {})
            if skeptical_q:
                skeptical_result = self.dispatcher.dispatch("web_search", {"query": skeptical_q})
                ctx.total_tool_calls += 1
                ctx.skeptical_followup_done = True
                self.engine._log_runtime({
                    "type": "skeptical_search_followup",
                    "iteration": iter_idx,
                    "reason": skeptical_reason,
                    "query": skeptical_q[:220],
                    "ok": bool(skeptical_result.get("ok")) if isinstance(skeptical_result, dict) else False,
                })
                self._ingest_search_hits(ctx, skeptical_result if isinstance(skeptical_result, dict) else {})
                sanitized_sk = ToolDispatcher.compact_tool_result("web_search", skeptical_result if isinstance(skeptical_result, dict) else {})
                messages.append({"role": "system", "content": "Tool result (web_search_skeptical):\n" + safe_json_dumps(sanitized_sk, ensure_ascii=False)[:2500]})
                ctx.last_tool_note = safe_json_dumps(sanitized_sk, ensure_ascii=False)[:800]

        if IntentClassifier.is_summary_request(last_user_text) and hit_count >= 3 and uniq_count >= 2:
            ctx.force_finalize_mode = True
        elif total_hits >= 3 and (new_hits / max(1, total_hits)) <= 0.34:
            ctx.force_finalize_mode = True
            messages.append({"role": "system", "content": "검색 결과가 충분히 수집되었습니다. 이제 추가 검색 없이 최종 답변을 작성하세요."})
        else:
            messages.append({"role": "system", "content": "검색 결과가 아직 불충분하거나 중복이 높습니다. 쿼리 관점을 바꿔 재계획 후 필요한 검색 1회를 수행하세요."})

    # ── Response Reflection ────────────────────────────────────────────────────
    def _build_reflection_prompt(self, ctx: ExecutionContext, user_query: str, final_answer: str) -> str | None:
        """Return a reflection prompt, or None if reflection should be skipped."""
        if ctx.prompt_class in self._REFLECTION_SKIP_CLASSES:
            return None
        if ctx.reflection_done:
            return None

        expected = set(self._REFLECTION_EXPECTED_TOOLS.get(ctx.prompt_class, set()))

        # Conditional: research_focused may also require write_file or schedule_task
        if ctx.prompt_class == "research_focused":
            low_q = user_query.lower()
            if any(kw in low_q for kw in ("저장", "save", "write", "파일로", "txt로")):
                expected.add("write_file")
            if any(kw in low_q for kw in ("매일", "매주", "매월", "schedule", "cron")):
                expected.add("schedule_task")

        if not expected:
            return None

        # Extract tool names from signature keys like "web_search:{...}"
        called_tools = set(sig.split(":", 1)[0] for sig in ctx.tool_signature_counts)

        # If all expected tools were called, no reflection needed
        if expected.issubset(called_tools):
            return None

        missing = expected - called_tools
        return (
            "REFLECTION CHECK (respond with ONLY 'OK' or 'RETRY: ...'):\n"
            f"- prompt_class: {ctx.prompt_class}\n"
            f"- expected_tools: {', '.join(sorted(expected))}\n"
            f"- actually_called: {', '.join(sorted(called_tools)) or '(none)'}\n"
            f"- missing_tools: {', '.join(sorted(missing))}\n"
            f"- user_question: {user_query[:300]}\n"
            f"- final_answer_preview: {final_answer[:300]}\n\n"
            "Is this response complete? Did the agent skip a required action?\n"
            "If the response is acceptable despite missing tools, reply: OK\n"
            "If a required tool was skipped, reply: RETRY: <one sentence: which tool to call and why>"
        )

    def _run_reflection(self, ctx: ExecutionContext, user_query: str, final_answer: str) -> str | None:
        """Run reflection check. Returns correction instruction or None if OK."""
        prompt = self._build_reflection_prompt(ctx, user_query, final_answer)
        if prompt is None:
            return None

        try:
            out = self.client.chat(
                messages=[{"role": "system", "content": prompt}],
                tools=[],
                timeout_sec=10.0,
                cancel_event=self.engine._cancel_event,
            )
            reply = (out.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        except RuntimeError as e:
            if str(e) == "cancelled_by_client":
                return None
            return None  # reflection failure should never block the response
        except Exception:
            return None  # reflection failure should never block the response

        self.engine._log_runtime({
            "type": "response_reflection",
            "prompt_class": ctx.prompt_class,
            "called_tools": list(ctx.tool_signature_counts.keys()),
            "missing_tools": list((set(self._REFLECTION_EXPECTED_TOOLS.get(ctx.prompt_class, set())) -
                                   set(sig.split("(")[0] for sig in ctx.tool_signature_counts))),
            "reflection_reply": reply[:200],
        })

        if reply.upper().startswith("OK"):
            return None
        return reply

    def _run_global_skepticism(self, ctx: ExecutionContext, user_query: str, final_answer: str) -> tuple[str, str, bool]:
        """Multi-persona self-critique pass for all final conclusions.

        Returns: (possibly_revised_answer, verdict, revised_flag)
        """
        if ctx.prompt_class in {"factcheck_focused", "workspace_ops"}:
            return final_answer, "skip_latency_sensitive_class", False
        if not final_answer.strip():
            return final_answer, "skip_empty", False
        called_tools = sorted(set(sig.split(":", 1)[0] for sig in ctx.tool_signature_counts.keys()))
        persona_verdicts: list[dict[str, Any]] = []
        for persona in self._SKEPTIC_PERSONAS:
            skeptic_prompt = (
                "You are a strict response auditor.\n"
                f"Persona: {persona['name']} ({persona['id']})\n"
                f"Focus: {persona['focus']}\n"
                "Review the draft answer and challenge it from this persona's perspective.\n"
                "If there is a meaningful issue, produce a corrected final answer.\n"
                "If acceptable, keep it.\n\n"
                "Output JSON ONLY:\n"
                "{\n"
                '  "persona_id": "string",\n'
                '  "verdict": "ok" | "revise",\n'
                '  "reason": "short reason",\n'
                '  "revised_answer": "full final answer text"\n'
                "}\n\n"
                f"prompt_class: {ctx.prompt_class}\n"
                f"user_query: {user_query[:600]}\n"
                f"called_tools: {', '.join(called_tools) if called_tools else '(none)'}\n"
                f"last_tool_note: {ctx.last_tool_note[:900] if ctx.last_tool_note else '(none)'}\n"
                f"draft_answer: {final_answer[:2400]}\n"
            )
            try:
                out = self.client.chat(
                    messages=[{"role": "system", "content": skeptic_prompt}],
                    tools=[],
                    timeout_sec=8.0,
                    cancel_event=self.engine._cancel_event,
                )
                raw = (out.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
                blocks = ToolCallParser.extract_json_blocks(raw)
                parsed = None
                for blk in blocks:
                    try:
                        obj = json.loads(blk)
                        if isinstance(obj, dict):
                            parsed = obj
                            break
                    except Exception:
                        continue
                if not isinstance(parsed, dict):
                    persona_verdicts.append({"persona_id": persona["id"], "verdict": "ok", "reason": "parse_failed", "revised_answer": ""})
                    continue
                pid = str(parsed.get("persona_id") or persona["id"]).strip() or persona["id"]
                verdict = str(parsed.get("verdict") or "ok").strip().lower()
                if verdict not in {"ok", "revise"}:
                    verdict = "ok"
                reason = str(parsed.get("reason") or "").strip()[:240]
                revised_answer = str(parsed.get("revised_answer") or "").strip()
                persona_verdicts.append({
                    "persona_id": pid,
                    "verdict": verdict,
                    "reason": reason,
                    "revised_answer": revised_answer,
                })
            except Exception as e:
                persona_verdicts.append({"persona_id": persona["id"], "verdict": "ok", "reason": f"error:{type(e).__name__}", "revised_answer": ""})

        revise_candidates = [p for p in persona_verdicts if p.get("verdict") == "revise" and p.get("revised_answer")]
        if not revise_candidates:
            verdict_summary = "ok:" + ",".join(f"{p.get('persona_id')}={p.get('reason') or 'ok'}" for p in persona_verdicts)
            return final_answer, verdict_summary[:380], False

        # Pick the strongest candidate: longest non-trivial revised answer.
        revise_candidates.sort(key=lambda x: len(str(x.get("revised_answer") or "")), reverse=True)
        chosen = revise_candidates[0]
        revised_answer = str(chosen.get("revised_answer") or "").strip()
        if not revised_answer or revised_answer == final_answer:
            verdict_summary = "ok:no_effective_revision"
            return final_answer, verdict_summary, False
        verdict_summary = "revise:" + ",".join(f"{p.get('persona_id')}={p.get('reason') or p.get('verdict')}" for p in persona_verdicts)
        return revised_answer, verdict_summary[:380], True

    def execute_task(self, messages: list[dict[str, Any]], source: str, user_id: str, user_original_text: str = "") -> str:
        ctx = self._init_context(messages, user_original_text)

        if ctx.prompt_class == "local_time":
            return self._handle_local_time_fastpath(source, user_id)

        for iter_idx in range(1, self.config.max_tool_iterations + 1):
            # Cooperative cancellation — if IPC client signalled cancel, exit
            # the iteration loop early rather than keep spinning reflection.
            if self.engine._cancel_event.is_set():
                self.engine._log_runtime({
                    "type": "cancelled",
                    "iteration": iter_idx,
                    "tool_calls": ctx.total_tool_calls,
                })
                return "[cancelled by client]"

            self._manage_context(messages, ctx, iter_idx)

            # Forced bootstrap on first iterations
            if self._try_forced_bootstrap(messages, ctx, iter_idx):
                continue

            # LLM call
            current_tools = [] if ctx.force_finalize_mode else ctx.selected_tools
            if ctx.force_finalize_mode and not ctx.finalize_notice_pushed:
                messages.append({"role": "system", "content": "수집된 정보가 답변 작성에 충분합니다. 이제 추가 도구 호출 없이 최종 답변을 작성하세요."})
                ctx.finalize_notice_pushed = True

            # Code/skill generation tasks need longer timeout for LLM to produce full code
            effective_timeout = float(LLM_QUERY_TIMEOUT_SEC)
            if ctx.prompt_class in ("coding_ops", "skill_ops"):
                effective_timeout = float(LLM_QUERY_TIMEOUT_MAX_SEC) * 2  # 80s for code gen
            try:
                out = self.client.chat(
                    messages=messages,
                    tools=current_tools,
                    timeout_sec=effective_timeout,
                    cancel_event=self.engine._cancel_event,
                )
            except RuntimeError as e:
                if str(e) == "cancelled_by_client" or self.engine._cancel_event.is_set():
                    self.engine._log_runtime({
                        "type": "cancelled",
                        "iteration": iter_idx,
                        "tool_calls": ctx.total_tool_calls,
                        "where": "llm_chat",
                    })
                    return "[cancelled by client]"
                raise
            message = out.get("choices", [{}])[0].get("message", {})
            tool_calls = message.get("tool_calls") or []
            text = (message.get("content") or "").strip()

            # Parse fallback tool calls from text
            if not tool_calls and text:
                late_loop_phase = iter_idx >= max(3, self.config.max_tool_iterations // 2 + 1)
                if IntentClassifier.is_context_reask_text(text) and (len(ctx.seen_search_urls) > 0 or ctx.total_tool_calls >= 2):
                    ctx.context_reask_streak += 1
                    if late_loop_phase and ctx.context_reask_streak >= 2 and ctx.total_tool_calls >= 3:
                        ctx.force_finalize_mode = True
                        if not ctx.loop_guard_pushed:
                            messages.append({"role": "system", "content": "LOOP GUARDRAIL:\n사용자 요청은 이미 충분히 구체적입니다. 이전 검색 주제를 다시 묻지 말고, 현재까지 수집한 결과를 바탕으로 불확실한 점을 명시하여 최종 답변을 작성하세요. 추가 도구 호출은 금지합니다."})
                            ctx.loop_guard_pushed = True
                        self.engine._log_runtime({"type": "loop_guardrail", "iteration": iter_idx, "reason": "repeated_context_reask_without_progress", "tool_calls": ctx.total_tool_calls, "seen_search_urls": len(ctx.seen_search_urls)})
                        continue
                else:
                    ctx.context_reask_streak = 0

                fallback_calls, text = ToolCallParser.parse_fallback_from_text(text)
                if fallback_calls:
                    tool_calls.extend(fallback_calls)

            if tool_calls:
                tool_calls = ToolCallParser.normalize_tool_calls(tool_calls)

            if tool_calls:
                self._handle_tool_calls(messages, ctx, tool_calls, text, iter_idx)
                continue

            # Final response (no tool calls)
            if not text:
                text = "정보가 부족해 결론을 완성하지 못했습니다. 필요한 추가 맥락을 한 가지 더 알려주세요."
                if ctx.last_tool_note:
                    text += f"\n\n최근 도구 결과: {ctx.last_tool_note}"

            user_query = self.engine._latest_user_query(messages)

            # ── Response Reflection (at most once) ───────────────────────────
            if not ctx.reflection_done:
                correction = self._run_reflection(ctx, user_query, text)
                ctx.reflection_done = True
                if correction:
                    # Identify the missing tool(s) for an explicit directive
                    expected = set(self._REFLECTION_EXPECTED_TOOLS.get(ctx.prompt_class, set()))
                    if ctx.prompt_class == "research_focused":
                        low_q = (user_query or "").lower()
                        if any(kw in low_q for kw in ("저장", "save", "write", "파일로", "txt로")):
                            expected.add("write_file")
                        if any(kw in low_q for kw in ("매일", "매주", "매월", "schedule", "cron")):
                            expected.add("schedule_task")
                    called_tools = set(sig.split(":", 1)[0] for sig in ctx.tool_signature_counts)
                    missing = list(expected - called_tools)
                    missing_str = ", ".join(missing) if missing else "(see correction)"

                    messages.append({"role": "system", "content": (
                        f"REFLECTION CORRECTION — your previous response is INCOMPLETE.\n\n"
                        f"USER REQUEST: {user_query}\n\n"
                        f"PROBLEM: You did NOT call the required tool(s): {missing_str}\n"
                        f"Reviewer note: {correction}\n\n"
                        f"REQUIRED ACTION: Immediately call {missing_str} with appropriate arguments. "
                        f"Do NOT produce a text-only response. Do NOT explain. Do NOT apologize. "
                        f"Just call the missing tool(s) with the right args based on the user request and the data already gathered.\n"
                        f"After the tool call(s) succeed, write a short final response confirming what was done."
                    )})
                    ctx.force_finalize_mode = False
                    self.engine._log_runtime({
                        "type": "reflection_retry",
                        "prompt_class": ctx.prompt_class,
                        "missing_tools": missing,
                        "correction": correction[:300],
                        "iteration": iter_idx,
                    })
                    continue  # re-enter loop for one more iteration

            # ── Global Skepticism Pass (exactly once) ───────────────────────
            if not ctx.skepticism_done:
                revised_text, verdict, revised = self._run_global_skepticism(ctx, user_query, text)
                ctx.skepticism_done = True
                text = revised_text
                self.engine._log_runtime({
                    "type": "skepticism_pass",
                    "prompt_class": ctx.prompt_class,
                    "personas": [p["id"] for p in self._SKEPTIC_PERSONAS],
                    "verdict": verdict,
                    "revised": revised,
                    "iteration": iter_idx,
                })

            self.engine._log_runtime({"type": "response_complete", "source": source, "user_id": user_id, "iterations": iter_idx, "tool_calls": ctx.total_tool_calls, "compression_events": ctx.compression_events})
            return text

        # Iteration limit reached
        fallback = "반복 한도에 도달했습니다. 요청 범위를 좁혀서 다시 시도해주세요."
        if ctx.last_tool_note:
            fallback += f"\n\n최근 도구 결과: {ctx.last_tool_note}"
        self.engine._log_runtime({"type": "response_fallback", "source": source, "user_id": user_id, "iterations": self.config.max_tool_iterations, "tool_calls": ctx.total_tool_calls, "compression_events": ctx.compression_events})
        return fallback


# ═══════════════════════════════════════════════════════════════════════════════
# Section 13: AgentEngine (thin orchestrator)
# ═══════════════════════════════════════════════════════════════════════════════

class AgentEngine:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.secrets = SecretStore(config.workspace, config.agent_name)
        # Resolve @secret: references in config
        self._resolve_config_secrets()
        self.memory = MemoryStore(config.workspace)
        self.tools = WorkspaceTools(config.workspace, self.memory, brave_api_key=config.brave_api_key, secret_store=self.secrets)
        self.client = LLMClient(
            config.api_base, config.api_key, config.models, config.request_timeout_sec,
            fallback_routes=config.fallback_routes, cooldown_sec=config.route_cooldown_sec,
            state_path=config.workspace / "logs" / "llm_route_state.json",
        )
        self._lock = threading.Lock()
        self.runtime_log_path = config.workspace / "logs" / "runtime_log.jsonl"
        self.runtime_log_path.parent.mkdir(parents=True, exist_ok=True)
        # Composed components
        self.context_mgr = ContextManager(config)
        self.prompt_builder = PromptBuilder(config, self.memory)
        self.dispatcher = ToolDispatcher(self.tools, self._log_runtime, config, client_ref=self.client)
        self._pending_restart_agent: str | None = None
        # Conversation continuity: remember last turn for follow-up reclassification
        self._last_prompt_class: str = ""
        self._last_user_query: str = ""
        # Cooperative cancellation (set by IPCRunner when a cancel request
        # arrives mid-respond). Checked at safe points in execute_task.
        self._cancel_event: threading.Event = threading.Event()
        self._bootstrap_identity_memory()

    def _resolve_config_secrets(self) -> None:
        """Resolve @secret: references in RuntimeConfig fields."""
        if SecretStore.is_ref(self.config.api_key):
            self.config.api_key = self.secrets.load(SecretStore.ref_key(self.config.api_key)) or ""
        if SecretStore.is_ref(self.config.telegram_token):
            self.config.telegram_token = self.secrets.load(SecretStore.ref_key(self.config.telegram_token)) or ""
        if SecretStore.is_ref(self.config.brave_api_key):
            self.config.brave_api_key = self.secrets.load(SecretStore.ref_key(self.config.brave_api_key)) or ""
        resolved_routes: list[dict[str, str]] = []
        for route in self.config.fallback_routes:
            r = dict(route)
            if SecretStore.is_ref(r.get("api_key", "")):
                r["api_key"] = self.secrets.load(SecretStore.ref_key(r["api_key"])) or ""
            resolved_routes.append(r)
        self.config.fallback_routes = resolved_routes

    def _bootstrap_identity_memory(self) -> None:
        fact = f"Agent identity: my name is {self.config.agent_name}."
        if not self.memory.has_memory_fact(fact, layer="L1"):
            self.memory.remember_fact(fact=fact, layer="L1", relation="identity", entity="agent", target_entity=self.config.agent_name)
        # Store aliases as L1 identity facts
        for alias in self.config.aliases:
            alias_fact = f"Agent identity: I am also known as '{alias}'. When users say '{alias}', they are referring to me."
            if not self.memory.has_memory_fact(alias, layer="L1"):
                self.memory.remember_fact(fact=alias_fact, layer="L1", relation="identity", entity="agent", target_entity=alias)

    def _log_runtime(self, payload: dict[str, Any]) -> None:
        row = {"ts": utc_now()}
        for k, v in payload.items():
            if isinstance(v, str):
                row[k] = safe_str(v)
            elif isinstance(v, list):
                row[k] = [safe_str(x) if isinstance(x, str) else x for x in v]
            else:
                row[k] = v
        with self.runtime_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _latest_user_query(self, messages: list[dict[str, Any]]) -> str:
        for msg in reversed(messages):
            if str(msg.get("role")) != "user":
                continue
            text = str(msg.get("content") or "").strip()
            m = re.match(r"^\[SubTask\s+\d+/\d+\]:\s*(.+)$", text)
            if m:
                text = m.group(1).strip()
            if text:
                return text
        return ""

    def _local_fastpath_response(self, user_text: str, prompt_class: str) -> str | None:
        if prompt_class == "direct_dialog" and IntentClassifier.is_simple_dialog_query(user_text):
            display_name = self.config.aliases[0] if self.config.aliases else self.config.agent_name
            return (
                f"안녕하세요. 저는 {display_name}({self.config.agent_name})입니다. "
                "PLANARIA_ONE 에이전트로 요청을 도와드릴 수 있어요."
            )
        if IntentClassifier.is_capability_overview_query(user_text):
            lines = [
                "제가 바로 도와드릴 수 있는 대표 기능입니다.",
                "1) 웹 검색 기반 최신 정보 요약",
                "2) 파일 읽기/쓰기/목록 조회",
                "3) 메모 저장/회상",
                "4) 예약 작업 등록/조회/취소",
            ]
            cfg = safe_json_load(self.config.workspace / "config.json", {})
            has_srt = bool(cfg.get("connectors", {}).get("srt_id"))
            if has_srt:
                lines.append("5) SRT 조회/예약")
            lines.append("원하시는 작업을 자연어로 바로 말씀해 주세요.")
            return "\n".join(lines)
        if prompt_class == "local_time" or IntentClassifier.is_local_time_query(user_text):
            kst = datetime.now(timezone(timedelta(hours=9)))
            utc = datetime.now(timezone.utc)
            return (
                f"현재 시간은 KST {kst.strftime('%Y-%m-%d %H:%M:%S')}입니다. "
                f"(UTC {utc.strftime('%Y-%m-%d %H:%M:%S')})"
            )
        return None

    def _select_tools_for_query(self, user_text: str) -> tuple[list[dict[str, Any]], list[str], str, dict[str, Any]]:
        query = (user_text or "").strip().lower()
        all_tools = self.tools.get_all_tools()
        all_names: list[str] = []
        for spec in all_tools:
            fn = spec.get("function", {}) if isinstance(spec, dict) else {}
            name = str(fn.get("name") or "")
            if name:
                all_names.append(name)

        if IntentClassifier.is_capability_overview_query(user_text):
            return [], [], "direct_dialog", {"reason": "capability_overview_fastpath", "scores": {"direct_dialog": 1}}
        if IntentClassifier.is_memory_recall_query(user_text):
            # Keep memory recall away from short dialog fast-path.
            return (
                [s for s in all_tools if str(s.get("function", {}).get("name", "")) in {"search_memory", "remember_fact"}],
                [n for n in all_names if n in {"search_memory", "remember_fact"}],
                "memory_focused",
                {"reason": "memory_recall_priority", "scores": {"memory_focused": 1}},
            )
        if (
            IntentClassifier.is_memory_store_query(user_text)
            and not IntentClassifier.should_force_web_grounding(user_text)
        ):
            # Pure-save intents (e.g., "내 이름은 지민이야. 기억해둬.") get a
            # tight memory-only tool slice so the model cannot wander off into
            # web_search loops. Combined "검색해서 기억해둬" requests fall
            # through to the general classifier, which keeps web_search in
            # scope while the memory_focused prompt still teaches the model
            # to finish with remember_fact.
            return (
                [s for s in all_tools if str(s.get("function", {}).get("name", "")) in {"remember_fact", "search_memory"}],
                [n for n in all_names if n in {"remember_fact", "search_memory"}],
                "memory_focused",
                {"reason": "memory_store_priority", "scores": {"memory_focused": 1}},
            )
        if IntentClassifier.is_simple_dialog_query(user_text):
            return [], [], "direct_dialog", {"reason": "short_social_dialog", "scores": {"direct_dialog": 1}}
        if IntentClassifier.is_local_time_query(user_text):
            return [], [], "local_time", {"reason": "local_time_query_no_web_search", "scores": {"local_time": 1}}

        intent_keywords: dict[str, list[str]] = {
            "research_focused": ["검색", "최신", "뉴스", "현황", "동향", "trend", "latest", "news", "search", "web", "근황"],
            "factcheck_focused": ["사실", "팩트", "검증", "근거", "출처", "fact", "verify", "evidence", "citation"],
            "memory_focused": ["기억", "기억나", "기억해", "memory", "remember", "과거 대화"],
            "workspace_ops": ["파일", "폴더", "디렉터리", "read", "write", "file", "folder", "shell", "명령", "현재 폴더"],
            "coding_ops": ["코드", "구현", "버그", "리팩토링", "테스트", "python", "javascript", "script", "debug", "coding"],
            "schedule_ops": ["일정", "스케줄", "cron", "schedule", "매일", "매주", "매월", "알림", "캘린더", "calendar", "오전", "오후", "아침", "저녁", "시에", "예약된", "예약 작업", "스케줄 보여", "예약 취소", "작업 취소", "scheduled task", "취소해", "삭제해", "지워", "list_scheduled", "cancel_scheduled"],
            "email_ops": ["이메일", "메일", "email", "mail", "inbox", "받은편지", "gmail"],
            "skill_ops": ["스킬", "skill", "install", "run skill", "허브", "clawhub", "srt", "예약", "기차표", "열차"],
            "config_ops": ["api key", "설정", "config", "토큰", "키 등록", "update_config", "등록", "바꾸", "변경", "apikey", "brave"],
        }
        scores: dict[str, int] = {}
        reasons: dict[str, list[str]] = {}
        for cls, kws in intent_keywords.items():
            hit: list[str] = []
            for kw in kws:
                if kw in query:
                    hit.append(kw)
            scores[cls] = len(hit)
            if hit:
                reasons[cls] = hit

        priority = ["factcheck_focused", "research_focused", "schedule_ops", "email_ops", "workspace_ops", "coding_ops", "memory_focused", "skill_ops", "config_ops"]
        prompt_class = "analysis_general"
        best_score = 0
        for cls in priority:
            score = int(scores.get(cls, 0))
            if score > best_score:
                best_score = score
                prompt_class = cls

        # Heuristic override: "검색 + 기억" mixed query should remain research-first.
        has_research_phrase = any(k in query for k in ["검색", "latest", "news", "트렌드", "trend", "동향", "최신", "뉴스"])
        has_memory_phrase = any(k in query for k in ["기억", "remember"])
        if has_research_phrase and has_memory_phrase:
            prompt_class = "research_focused"
            reasons.setdefault("heuristic_override", []).append("research_plus_memory -> research_focused")

        class_tools: dict[str, set[str]] = {
            "research_focused": {"web_search", "search_hub", "search_memory"},
            "factcheck_focused": {"web_search", "search_memory"},
            "memory_focused": {"search_memory", "remember_fact"},
            "workspace_ops": {"list_files", "read_file", "write_file", "run_shell"},
            "coding_ops": {"list_files", "read_file", "write_file", "run_shell", "search_hub", "web_search"},
            "schedule_ops": {"schedule_task", "list_scheduled_tasks", "cancel_scheduled_task", "check_calendar"},
            "email_ops": {"check_email", "search_memory"},
            "skill_ops": {"install_skill", "run_skill", "search_hub", "web_search"},
            "config_ops": {"update_config", "web_search"},
            "analysis_general": {"web_search", "search_memory", "list_files", "read_file"},
        }
        selected = set(class_tools.get(prompt_class, set()))
        if best_score == 0:
            reasons[prompt_class] = ["no_strong_keyword_match_use_safe_default"]

        # Cross-intent augmentation: add tools from secondary intents
        if any(k in query for k in ["저장", "파일", "txt", "write", "save"]):
            selected.update({"write_file", "read_file", "list_files"})
            reasons.setdefault("cross_intent", []).append("file_ops_detected")
        if any(k in query for k in ["매일", "매주", "매월", "cron", "schedule", "자동으로"]):
            selected.add("schedule_task")
            reasons.setdefault("cross_intent", []).append("schedule_detected")
        if any(k in query for k in ["기억", "remember"]):
            selected.add("remember_fact")
            reasons.setdefault("cross_intent", []).append("memory_detected")

        # SRT-intent should prefer skill_ops over generic schedule routing.
        srt_signals = ("srt", "수서", "부산", "대전", "열차", "기차")
        has_srt_tool = any(n == "srt_reserve" for n in all_names)
        if has_srt_tool and any(k in query for k in srt_signals):
            prompt_class = "skill_ops"
            selected = set(class_tools.get("skill_ops", set()))
            selected.add("srt_reserve")
            reasons.setdefault("heuristic_override", []).append("srt_query -> skill_ops")

        # When routed to skill_ops, include ALL external skills so the LLM knows about them
        if prompt_class == "skill_ops":
            for name in all_names:
                if name not in INTERNAL_TOOL_NAMES:
                    selected.add(name)
                    reasons.setdefault("auto_included_skills", []).append(name)

        for name in all_names:
            if name in INTERNAL_TOOL_NAMES:
                continue
            if name.lower() in query:
                selected.add(name)
                reasons.setdefault("dynamic_skill_ops", []).append(name)

        selected.discard("reset_workspace")

        excluded_by_constraint = IntentClassifier.parse_tool_prohibitions(query, all_names)
        if excluded_by_constraint:
            for tool_name in excluded_by_constraint:
                selected.discard(tool_name)
            reasons["excluded_by_user_constraint"] = list(excluded_by_constraint)

        ordered_specs: list[dict[str, Any]] = []
        ordered_names: list[str] = []
        for spec in all_tools:
            fn = spec.get("function", {}) if isinstance(spec, dict) else {}
            name = str(fn.get("name") or "")
            if name in selected:
                ordered_specs.append(spec)
                ordered_names.append(name)

        route_meta = {"reason": "keyword_routing", "scores": scores, "matched_keywords": reasons, "best_score": best_score}
        return ordered_specs, ordered_names, prompt_class, route_meta

    def _plan_needs_web_replan(self, user_text: str, job_plan: dict[str, Any]) -> bool:
        tasks_raw = job_plan.get("tasks", []) if isinstance(job_plan, dict) else []
        constraints_raw = job_plan.get("constraints", []) if isinstance(job_plan, dict) else []
        tasks = [str(x).strip() for x in tasks_raw if str(x).strip()] if isinstance(tasks_raw, list) else []
        constraints = [str(x).strip() for x in constraints_raw if str(x).strip()] if isinstance(constraints_raw, list) else []
        if IntentClassifier.is_memory_recall_query(user_text):
            return False
        if not tasks:
            return True
        if len(tasks) == 1 and tasks[0].lower() == (user_text or "").strip().lower():
            return True
        joined = " ".join(tasks + constraints).lower()
        refusal_hints = ["cannot", "can't", "unable", "not possible", "불가", "불가능", "알려줄 수 없", "제공할 수 없", "check your device", "가상 비서", "직접 확인"]
        if any(h in joined for h in refusal_hints):
            return True
        if IntentClassifier.should_force_web_grounding(user_text):
            has_search_step = any(("search" in t.lower()) or ("검색" in t) or ("근거" in t) or ("출처" in t) for t in tasks)
            if not has_search_step:
                return True
        return False

    def _summarize_search_for_planning(self, search_result: dict[str, Any], max_items: int = 5) -> str:
        if not isinstance(search_result, dict) or not search_result.get("ok"):
            return ""
        rows = search_result.get("results", [])
        if not isinstance(rows, list) or not rows:
            return ""
        lines = []
        source = str(search_result.get("source") or "unknown")
        lines.append(f"- source: {source}")
        for i, row in enumerate(rows[:max_items], start=1):
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or row.get("name") or "").strip()
            snippet = str(row.get("snippet") or row.get("desc") or "").strip()
            url = str(row.get("url") or "").strip()
            if title:
                lines.append(f"- [{i}] title: {title[:140]}")
            if snippet:
                lines.append(f"  snippet: {snippet[:220]}")
            if url:
                lines.append(f"  url: {url[:240]}")
        return "\n".join(lines)

    def _auto_extract_and_store_facts(self, user_text: str, assistant_text: str, source: str, user_id: str) -> None:
        if len(user_text.strip()) < 10 and len(assistant_text.strip()) < 30:
            return
        if IntentClassifier.is_local_time_query(user_text):
            return
        if IntentClassifier.is_memory_recall_query(user_text):
            return
        user_part = user_text.strip()[:600]
        assistant_part = assistant_text.strip()[:1200]
        extraction_prompt = (
            "You are a knowledge graph extractor. Analyze the conversation turn below and extract "
            "important facts worth remembering for future conversations.\n\n"
            "Output a JSON array of fact objects. Each object must have:\n"
            "- \"fact\": A clear, self-contained statement of the fact\n"
            "- \"layer\": Classification (STRICT rules):\n"
            "    L1 = Agent identity ONLY (name, personality, core behavior). NEVER use L1 for anything else.\n"
            "    L2 = User profile, preferences, ongoing projects, goals, relationships between entities\n"
            "    L3 = Episodic/session-specific facts, task results, transient discoveries\n"
            "- \"entity\": The subject of the fact (e.g., 'user', 'agent', a project name, a person name)\n"
            "- \"relation\": Relationship type (e.g., 'preference', 'has_project', 'is_type', 'located_in', "
            "'works_on', 'goal', 'discovered', 'status')\n"
            "- \"target_entity\": The object entity if linking two entities, otherwise empty string\n\n"
            "Rules:\n"
            "1) Only extract facts worth remembering LONG-TERM. Skip transient queries, greetings, "
            "filler, and information that is only relevant to this specific conversation turn.\n"
            "2) Extract facts from BOTH user and assistant messages. User messages reveal preferences, "
            "projects, goals. Assistant messages may contain discovered facts worth caching.\n"
            "3) Be concise. Each fact should be 1-2 sentences max.\n"
            "4) Do NOT extract the raw query itself as a fact.\n"
            "5) DO extract user-revealed information.\n"
            "6) DO extract discovered knowledge worth caching.\n"
            "7) Maximum 4 facts per turn. If nothing is worth remembering, return empty array [].\n\n"
            f"User message:\n{user_part}\n\n"
            f"Assistant response:\n{assistant_part}\n\n"
            "Output JSON array only, no markdown:"
        )
        try:
            out = self.client.chat(
                [{"role": "system", "content": "Extract knowledge graph facts. Output raw JSON array only."}, {"role": "user", "content": extraction_prompt}],
                timeout_sec=float(LLM_QUERY_TIMEOUT_SEC),
                cancel_event=self._cancel_event,
            )
            text = out.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            start = text.find("[")
            end = text.rfind("]")
            if start == -1 or end == -1:
                return
            facts_raw = json.loads(text[start : end + 1])
            if not isinstance(facts_raw, list):
                return
            saved: list[dict[str, str]] = []
            for item in facts_raw[:4]:
                if not isinstance(item, dict):
                    continue
                fact = str(item.get("fact", "")).strip()
                layer = str(item.get("layer", "L3")).strip().upper()
                entity = str(item.get("entity", "user")).strip()
                relation = str(item.get("relation", "related_to")).strip()
                target_entity = str(item.get("target_entity", "")).strip()
                if not fact or len(fact) < 5:
                    continue
                result = self.memory.remember_fact(fact=fact, layer=layer, relation=relation, entity=entity, target_entity=target_entity)
                if result.get("note") != "duplicate_skipped":
                    saved.append({"fact": fact, "layer": result.get("layer", layer), "entity": entity})
            if saved:
                self._log_runtime({"type": "auto_graph_extraction", "source": source, "user_id": user_id, "count": len(saved), "facts": saved[:4]})
        except RuntimeError as e:
            if str(e) == "cancelled_by_client":
                self._log_runtime({"type": "auto_graph_extraction_cancelled"})
                return
            self._log_runtime({"type": "auto_graph_extraction_error", "error": f"{type(e).__name__}: {safe_str(e)}"})
        except Exception as e:
            self._log_runtime({"type": "auto_graph_extraction_error", "error": f"{type(e).__name__}: {safe_str(e)}"})

    def _synthesize_task_outputs(self, user_text: str, task_outputs: list[str]) -> str:
        combined_raw = "\n\n---\n\n".join(f"[Raw Task {i+1}]:\n{out}" for i, out in enumerate(task_outputs))
        if len(combined_raw) > 12000:
            combined_raw = combined_raw[:12000] + "\n...[truncated]"
        synthesis_prompt = (
            "You are a response editor. The user asked a question and multiple sub-tasks produced raw outputs.\n"
            "Your job is to merge them into ONE clean, coherent response.\n\n"
            "Rules:\n"
            "1) REMOVE all duplicate information.\n"
            "2) REMOVE content not relevant to the user's original question.\n"
            "3) Do NOT include [Task N] labels.\n"
            "4) Organize logically with clear headings if needed.\n"
            "5) Preserve all source citations and factual details.\n"
            "6) Respond in the same language as the user's question.\n"
            "7) Do NOT add new information.\n\n"
            f"User's original question: {user_text}\n\n"
            f"Raw task outputs to merge:\n{combined_raw}"
        )
        try:
            out = self.client.chat(
                [{"role": "system", "content": "You are a response editor that merges and deduplicates multi-task outputs."}, {"role": "user", "content": synthesis_prompt}],
                timeout_sec=float(LLM_QUERY_TIMEOUT_SEC),
                cancel_event=self._cancel_event,
            )
            synthesized = out.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if synthesized and len(synthesized) > 50:
                self._log_runtime({"type": "task_synthesis_complete", "input_tasks": len(task_outputs), "output_chars": len(synthesized)})
                return synthesized
        except RuntimeError as e:
            if str(e) == "cancelled_by_client":
                self._log_runtime({"type": "task_synthesis_cancelled"})
                return "\n\n".join(task_outputs)
            self._log_runtime({"type": "task_synthesis_failed", "error": f"{type(e).__name__}: {safe_str(e)}"})
        except Exception as e:
            self._log_runtime({"type": "task_synthesis_failed", "error": f"{type(e).__name__}: {safe_str(e)}"})
        return "\n\n".join(task_outputs)

    # ── Onboarding: rule-based conversation before LLM is available ──

    def _onboarding_stage(self) -> str:
        """Return current onboarding stage from config. '' means completed."""
        cfg_path = self.config.workspace / "config.json"
        data = safe_json_load(cfg_path, {})
        return str(data.get("onboarding", {}).get("stage", ""))

    def _set_onboarding_stage(self, stage: str) -> None:
        cfg_path = self.config.workspace / "config.json"
        data = safe_json_load(cfg_path, {})
        if "onboarding" not in data:
            data["onboarding"] = {}
        data["onboarding"]["stage"] = stage
        write_json(cfg_path, data)

    @staticmethod
    def _brave_key_prompt() -> str:
        return (
            "웹 검색 설정\n\n"
            "Brave Search API 키를 입력해주세요.\n"
            "  (발급: https://brave.com/search/api/ → Get Started → Free 플랜)\n\n"
            "입력하지 않으면 DuckDuckGo를 사용하지만, 검색 품질이 떨어질 수 있습니다.\n"
            "건너뛰려면 skip을 입력하세요:"
        )

    @staticmethod
    def _integrations_menu() -> str:
        return (
            "추가 서비스를 연동하시겠습니까? (선택사항)\n\n"
            "  1) telegram — Telegram 봇 연동\n"
            "  2) gmail    — Gmail 이메일 조회/알림\n"
            "  3) calendar — Google Calendar 일정 조회/알림\n"
            "  4) srt      — SRT 열차 예약 (계정 연동)\n"
            "  5) skip     — 건너뛰기 (나중에 설정 가능)\n\n"
            "번호 또는 서비스 이름을 입력해주세요 (중간에 'back' 또는 'skip' 가능):"
        )

    def _save_integration_field(self, service: str, key: str, value: Any) -> None:
        cfg_path = self.config.workspace / "config.json"
        data = safe_json_load(cfg_path, {})
        if "integrations" not in data:
            data["integrations"] = {}
        if service not in data["integrations"]:
            data["integrations"][service] = {}
        data["integrations"][service][key] = value
        write_json(cfg_path, data)

    def _store_secret_and_ref(self, config_path_parts: list[str], secret_key: str, value: str) -> None:
        """Store value in SecretStore and save @secret: reference in config.json."""
        self.secrets.store(secret_key, value)
        cfg_path = self.config.workspace / "config.json"
        data = safe_json_load(cfg_path, {})
        # Navigate to parent dict and set the reference
        node = data
        for part in config_path_parts[:-1]:
            if part not in node:
                node[part] = {}
            node = node[part]
        node[config_path_parts[-1]] = SecretStore.make_ref(secret_key)
        write_json(cfg_path, data)

    def _llm_parse_integration_intent(self, user_text: str) -> str | None:
        """Use LLM to parse natural language into an integration service keyword.
        Returns one of: 'telegram', 'gmail', 'calendar', 'srt', 'skip', or None."""
        if not self.config.api_key:
            return None
        try:
            messages = [
                {"role": "system", "content": (
                    "You are a classifier. The user is choosing which service to integrate. "
                    "Available services: telegram, gmail, calendar, srt, skip. "
                    "Reply with ONLY the single keyword that best matches the user's intent. "
                    "If unclear, reply 'unknown'. No explanation."
                )},
                {"role": "user", "content": user_text},
            ]
            result = self.client.chat(messages, timeout_sec=10, cancel_event=self._cancel_event)
            answer = str(result.get("content") or "").strip().lower()
            if answer in ("telegram", "gmail", "calendar", "srt", "skip"):
                return answer
        except Exception:
            pass
        return None

    def _handle_onboarding(self, user_text: str) -> str | None:
        """Process onboarding input. Returns response string, or None if onboarding is done."""
        text = user_text.strip()
        has_primary = bool(self.config.api_key)
        has_fallback = bool(self.config.fallback_routes)
        stage = self._onboarding_stage()

        # Already completed onboarding and has API key → normal mode
        if stage == "done" and (has_primary or has_fallback):
            return None

        # Fresh workspace: no stage set yet → start onboarding
        if not stage:
            if has_primary or has_fallback:
                self._set_onboarding_stage("done")
                return None
            self._set_onboarding_stage("wait_name")
            return (
                "안녕하세요! 저는 PLANARIA 에이전트입니다.\n"
                "에이전트 이름을 정해주세요 (영문만, 예: fox, nova, jarvis):"
            )

        # Stage: agent name (English only, used as workspace directory)
        if stage == "wait_name":
            if not text:
                return "영문 이름을 입력해주세요 (예: fox, nova, jarvis):"
            name = re.sub(r'[^a-zA-Z0-9_]', '', text).strip().lower()
            if not name:
                return "영문 알파벳과 숫자만 사용할 수 있습니다. 다시 입력해주세요:"

            # Copy current workspace to new workspace if name differs
            current_ws = self.config.workspace
            base_dir = current_ws.parent
            new_ws = (base_dir / f"workspace_{name}").resolve()
            needs_restart = new_ws != current_ws

            if needs_restart and not new_ws.exists():
                shutil.copytree(str(current_ws), str(new_ws))

            # Update config in the target workspace
            target_ws = new_ws if needs_restart else current_ws
            cfg_path = target_ws / "config.json"
            data = safe_json_load(cfg_path, {})
            if "agent" not in data:
                data["agent"] = {}
            data["agent"]["name"] = name
            if "onboarding" not in data:
                data["onboarding"] = {}
            data["onboarding"]["stage"] = "wait_nickname"
            write_json(cfg_path, data)

            if needs_restart:
                self._pending_restart_agent = name
                return (
                    f"'{name}' 워크스페이스를 생성했습니다.\n"
                    f"새 워크스페이스(workspace_{name})로 재시작합니다..."
                )

            self.config.agent_name = name
            self._set_onboarding_stage("wait_nickname")
            return (
                f"이름: {name}\n"
                "닉네임도 지어줄까요? (한글 등 자유, 건너뛰려면 skip):"
            )

        # Stage: nickname (display name, any language)
        if stage == "wait_nickname":
            cfg_path = self.config.workspace / "config.json"
            data = safe_json_load(cfg_path, {})
            agent_name = data.get("agent", {}).get("name", self.config.agent_name)
            low = text.lower().strip() if text else ""

            if low in ("skip", "스킵", "건너뛰기", "패스", "no", ""):
                display = agent_name
            else:
                display = text.strip()

            if "agent" not in data:
                data["agent"] = {}
            data["agent"]["display_name"] = display
            if display != agent_name:
                aliases = data["agent"].get("aliases", [])
                if display not in aliases:
                    aliases.insert(0, display)
                data["agent"]["aliases"] = aliases
            write_json(cfg_path, data)
            self.config.aliases = data["agent"].get("aliases", [])

            # Store identity in memory
            fact = f"Agent identity: my name is {agent_name}."
            if not self.memory.has_memory_fact(fact, layer="L1"):
                self.memory.remember_fact(fact=fact, layer="L1", relation="identity", entity="agent", target_entity=agent_name)
            if display != agent_name:
                alias_fact = f"Agent identity: I am also known as '{display}'. When users say '{display}', they are referring to me."
                if not self.memory.has_memory_fact(display, layer="L1"):
                    self.memory.remember_fact(fact=alias_fact, layer="L1", relation="identity", entity="agent", target_entity=display)

            self._set_onboarding_stage("wait_llm_provider")
            notice = SECURITY_NOTICE.format(backend=self.secrets.backend_name)
            name_line = f"이름: {agent_name}" + (f", 닉네임: {display}" if display != agent_name else "")
            return (
                f"좋습니다! {name_line}\n\n"
                f"{notice}\n"
                "LLM 서비스를 선택해주세요:\n\n"
                "  1) openrouter  — 다양한 모델을 하나의 키로 (추천)\n"
                "  2) openai      — GPT 모델 (platform.openai.com)\n"
                "  3) anthropic   — Claude 모델 (console.anthropic.com)\n"
                "  4) google      — Gemini 모델 (aistudio.google.com)\n"
                "  5) upstage     — Solar Pro 3 (console.upstage.ai)\n"
                "  6) deepseek    — DeepSeek 모델 (platform.deepseek.com)\n\n"
                "번호 또는 이름을 입력해주세요:"
            )

        # ── LLM provider/service definitions ──
        _LLM_PROVIDERS: dict[str, dict[str, str]] = {
            "openrouter": {"api_base": "https://openrouter.ai/api/v1", "default_model": "openai/gpt-4o"},
            "openai":     {"api_base": "https://api.openai.com/v1", "default_model": "gpt-4o-mini"},
            "anthropic":  {"api_base": "https://api.anthropic.com/v1", "default_model": "claude-sonnet-4-20250514"},
            "google":     {"api_base": "https://generativelanguage.googleapis.com/v1beta/openai", "default_model": "gemini-2.0-flash"},
            "upstage":    {"api_base": "https://api.upstage.ai/v1/solar", "default_model": "solar-pro3-preview"},
            "deepseek":   {"api_base": "https://api.deepseek.com", "default_model": "deepseek-chat"},
        }
        _PROVIDER_ALIASES: dict[str, str] = {"1": "openrouter", "2": "openai", "3": "anthropic", "4": "google", "5": "upstage", "6": "deepseek"}

        # Stage: LLM provider selection
        if stage == "wait_llm_provider":
            if not text:
                return "번호 또는 서비스 이름을 입력해주세요:"
            low = text.lower().strip()
            # Hidden preset
            if low == "keti-conv":
                if not KETI_CONV_API_BASE:
                    return (
                        "keti-conv 프리셋은 환경변수 PLANARIA_KETI_CONV_API_BASE가 설정되어 있어야 사용할 수 있습니다.\n"
                        "다른 서비스를 선택하거나 관리자에게 문의하세요.\n"
                        "서비스 번호 또는 이름을 다시 입력해주세요:"
                    )
                self.secrets.store("llm_api_key", "keti-conv")
                cfg_path = self.config.workspace / "config.json"
                data = safe_json_load(cfg_path, {})
                if "llm" not in data:
                    data["llm"] = {}
                data["llm"]["models"] = [KETI_CONV_MODEL]
                data["llm"]["api_base"] = KETI_CONV_API_BASE
                data["llm"]["api_key"] = SecretStore.make_ref("llm_api_key")
                data["llm"]["context_max_chars"] = 2600
                data["llm"]["compressed_context_chars"] = 1800
                data["llm"]["max_tool_iterations"] = 8
                data["llm"]["route_cooldown_sec"] = 21600
                if "onboarding" not in data:
                    data["onboarding"] = {}
                data["onboarding"]["llm_provider"] = "keti-conv"
                write_json(cfg_path, data)
                self.config.api_key = "keti-conv"
                self.config.models = [KETI_CONV_MODEL]
                self.config.api_base = KETI_CONV_API_BASE
                self.client.api_key = "keti-conv"
                self.client.api_base = KETI_CONV_API_BASE
                self.client.models = self.config.models
                self._set_onboarding_stage("wait_keti_deepseek_key")
                return (
                    "KETI 내부 LLM 설정 중...\n"
                    "  Primary: Solar-Open-100B (내부 서버)\n\n"
                    "외부 Fallback으로 DeepSeek을 사용합니다.\n"
                    "DeepSeek API 키를 입력해주세요 (platform.deepseek.com에서 발급):"
                )
            # Resolve alias
            provider = _PROVIDER_ALIASES.get(low, low)
            if provider not in _LLM_PROVIDERS:
                return (
                    "알 수 없는 서비스입니다. 다시 선택해주세요:\n\n"
                    "  1) openrouter  2) openai  3) anthropic\n"
                    "  4) google  5) upstage  6) deepseek\n\n"
                    "번호 또는 이름을 입력해주세요:"
                )
            # Save chosen provider
            pinfo = _LLM_PROVIDERS[provider]
            cfg_path = self.config.workspace / "config.json"
            data = safe_json_load(cfg_path, {})
            if "llm" not in data:
                data["llm"] = {}
            data["llm"]["api_base"] = pinfo["api_base"]
            data["llm"]["models"] = [pinfo["default_model"]]
            if "onboarding" not in data:
                data["onboarding"] = {}
            data["onboarding"]["llm_provider"] = provider
            write_json(cfg_path, data)
            self.config.api_base = pinfo["api_base"]
            self.config.models = [pinfo["default_model"]]
            self.client.api_base = pinfo["api_base"]
            self.client.models = [pinfo["default_model"]]
            self._set_onboarding_stage("wait_api_key")
            return f"{provider} 선택 완료.\nAPI 키를 입력해주세요:"

        # Stage: API key input
        if stage == "wait_api_key":
            if len(text) >= 15 and " " not in text:
                # Verify the key before saving
                print("[onboarding] API 키 검증 중...")
                ok, msg = self.client.verify_key(self.config.api_base, text, self.config.models[0])
                if not ok:
                    return f"API 키 검증 실패: {msg}\n다시 입력해주세요:"
                self._store_secret_and_ref(["llm", "api_key"], "llm_api_key", text)
                self.config.api_key = text
                self.client.api_key = text
                cfg_path = self.config.workspace / "config.json"
                data = safe_json_load(cfg_path, {})
                chosen = data.get("onboarding", {}).get("llm_provider", "")
                # OpenRouter has options; others go straight to integrations
                if chosen == "openrouter":
                    self._set_onboarding_stage("wait_llm_options")
                    return (
                        "API 키 검증 성공!\n\n"
                        "모델 옵션을 선택해주세요:\n\n"
                        "  1) standard — 기본 모델 사용 (gpt-4o 등, 유료)\n"
                        "  2) free     — 무료 모델만 사용 (gemma, llama, deepseek 등)\n\n"
                        "번호 또는 이름을 입력해주세요:"
                    )
                else:
                    self._set_onboarding_stage("wait_brave_key")
                    return f"API 키 검증 성공! (서비스: {chosen})\n\n" + self._brave_key_prompt()
            else:
                return "API 키 형식이 올바르지 않습니다. 다시 입력해주세요:"

        # Stage: keti-conv DeepSeek fallback key
        if stage == "wait_keti_deepseek_key":
            if len(text) >= 15 and " " not in text:
                print("[onboarding] DeepSeek API 키 검증 중...")
                ok, msg = self.client.verify_key("https://api.deepseek.com", text, "deepseek-chat")
                if not ok:
                    return f"DeepSeek API 키 검증 실패: {msg}\n다시 입력해주세요:"
                self.secrets.store("deepseek_api_key", text)
                self._store_secret_and_ref(["llm", "fallback_routes_pending_key"], "deepseek_api_key", text)
                cfg_path = self.config.workspace / "config.json"
                data = safe_json_load(cfg_path, {})
                data.setdefault("llm", {})["fallback_routes"] = [
                    {"model": "deepseek-chat", "api_base": "https://api.deepseek.com", "api_key": SecretStore.make_ref("deepseek_api_key")}
                ]
                # Clean up temp field
                data["llm"].pop("fallback_routes_pending_key", None)
                write_json(cfg_path, data)
                self.config.fallback_routes = [
                    {"model": "deepseek-chat", "api_base": "https://api.deepseek.com", "api_key": text}
                ]
                self.client.fallback_routes = self.config.fallback_routes
                self._set_onboarding_stage("wait_brave_key")
                return (
                    "KETI 내부 LLM 설정 완료!\n"
                    "  Primary: Solar-Open-100B (내부 서버)\n"
                    "  Fallback: DeepSeek-Chat (외부)\n\n"
                    + self._brave_key_prompt()
                )
            else:
                return "API 키 형식이 올바르지 않습니다. 다시 입력해주세요:"

        # Stage: LLM options (currently for OpenRouter)
        if stage == "wait_llm_options":
            low = text.lower().strip()
            if low in ("1", "standard", "기본"):
                self._set_onboarding_stage("wait_brave_key")
                return "기본 모델(gpt-4o)이 설정되었습니다.\n\n" + self._brave_key_prompt()
            elif low in ("2", "free", "무료"):
                cfg_path = self.config.workspace / "config.json"
                data = safe_json_load(cfg_path, {})
                if "llm" not in data:
                    data["llm"] = {}
                free_models = [
                    "google/gemma-3-4b-it:free",
                    "meta-llama/llama-4-scout:free",
                    "deepseek/deepseek-chat-v3-0324:free",
                    "qwen/qwen3-4b:free",
                ]
                data["llm"]["models"] = free_models
                write_json(cfg_path, data)
                self.config.models = free_models
                self.client.models = free_models
                self._set_onboarding_stage("wait_brave_key")
                return (
                    "무료 모델이 설정되었습니다!\n"
                    "  사용 모델: " + ", ".join(m.split("/")[-1] for m in free_models) + "\n\n"
                    + self._brave_key_prompt()
                )
            else:
                return (
                    "다시 선택해주세요:\n\n"
                    "  1) standard — 기본 모델 (유료)\n"
                    "  2) free     — 무료 모델만 사용\n\n"
                    "번호 또는 이름을 입력해주세요:"
                )

        # Stage: Brave Search API key
        if stage == "wait_brave_key":
            low = text.lower().strip()
            if low in ("skip", "스킵", "건너뛰기", "패스", "no", ""):
                self._set_onboarding_stage("wait_integrations")
                return "Brave 검색 없이 진행합니다 (DuckDuckGo 사용).\n\n" + self._integrations_menu()
            # Brave keys typically start with "BSA"
            if len(text) >= 10 and " " not in text:
                print("[onboarding] Brave API 키 검증 중...")
                try:
                    resp = requests.get("https://api.search.brave.com/res/v1/web/search",
                                        headers={"Accept": "application/json", "X-Subscription-Token": text},
                                        params={"q": "test", "count": 1}, timeout=10)
                    if resp.status_code == 401 or resp.status_code == 403:
                        return f"Brave API 키 검증 실패 (HTTP {resp.status_code}). 다시 입력하거나 skip을 입력하세요:"
                except Exception:
                    pass  # network issue — accept the key anyway
                self._store_secret_and_ref(["search", "brave_api_key"], "brave_api_key", text)
                self.config.brave_api_key = text
                self.tools.brave_api_key = text
                self._set_onboarding_stage("wait_integrations")
                return "Brave Search API 키 검증 성공!\n\n" + self._integrations_menu()
            return "API 키 형식이 올바르지 않습니다. 다시 입력하거나 skip을 입력하세요:"

        # ── Global back/skip for any sub-stage ──
        _BACK_WORDS = ("back", "뒤로", "돌아가기", "이전", "취소", "cancel")
        _SKIP_WORDS = ("skip", "스킵", "건너뛰기", "패스")
        if text and stage.startswith("wait_"):
            low_cmd = text.lower().strip()
            # back → return to integration menu
            if low_cmd in _BACK_WORDS and stage not in ("wait_name", "wait_nickname", "wait_llm_provider", "wait_api_key", "wait_llm_options", "wait_brave_key", "wait_integrations"):
                self._set_onboarding_stage("wait_integrations")
                return self._integrations_menu()
            # skip → finish onboarding entirely
            if low_cmd in _SKIP_WORDS and stage not in ("wait_name", "wait_llm_provider", "wait_api_key"):
                if stage == "wait_integrations":
                    self._set_onboarding_stage("done")
                    return f"설정 완료! 저는 '{self.config.agent_name}', 당신의 AI 에이전트입니다.\n이제 무엇이든 물어보세요."
                self._set_onboarding_stage("wait_integrations")
                return self._integrations_menu()

        # Stage: integration service selection
        if stage == "wait_integrations":
            if not text:
                return self._integrations_menu()
            low = text.lower().strip()
            if low in ("skip", "건너뛰기", "스킵", "no", "아니오", "아니요", "패스", "done", "완료", "5"):
                self._set_onboarding_stage("done")
                return f"설정 완료! 저는 '{self.config.agent_name}', 당신의 AI 에이전트입니다.\n이제 무엇이든 물어보세요."
            if "telegram" in low or "텔레그램" in low or low == "1":
                self._set_onboarding_stage("wait_telegram_token")
                return (
                    "Telegram 봇을 설정합니다.\n"
                    "@BotFather에서 봇을 생성하고 토큰을 입력해주세요:\n"
                    "(예: 123456789:ABCdefGHI...)"
                )
            if "gmail" in low or "지메일" in low or low == "2":
                self._set_onboarding_stage("wait_gmail_email")
                return "Gmail을 설정합니다.\nGmail 주소를 입력해주세요:"
            if "calendar" in low or "캘린더" in low or low == "3":
                self._set_onboarding_stage("wait_gcal_creds")
                return (
                    "Google Calendar를 설정합니다.\n"
                    "Google Cloud Console에서 OAuth 2.0 credentials.json을 다운로드한 뒤\n"
                    "파일의 전체 경로를 입력해주세요:"
                )
            if "srt" in low or low == "4":
                self._set_onboarding_stage("wait_srt_id")
                return "SRT 계정을 설정합니다.\nSRT 멤버십 ID (휴대폰 번호)를 입력해주세요:"
            # LLM fallback: try to parse natural language intent
            parsed = self._llm_parse_integration_intent(text)
            if parsed:
                return self._handle_onboarding(parsed)
            return self._integrations_menu()

        # Sub-stages: Gmail setup
        if stage == "wait_gmail_email":
            if not text or "@" not in text:
                return "올바른 Gmail 주소를 입력해주세요:"
            self._save_integration_field("gmail", "username", text)
            self._save_integration_field("gmail", "imap_host", "imap.gmail.com")
            self._save_integration_field("gmail", "imap_port", 993)
            self._save_integration_field("gmail", "use_ssl", True)
            self._set_onboarding_stage("wait_gmail_apppass")
            return (
                f"Gmail 계정: {text}\n"
                "앱 비밀번호를 입력해주세요.\n"
                "(Google 계정 → 보안 → 2단계 인증 → 앱 비밀번호에서 생성)"
            )

        if stage == "wait_gmail_apppass":
            if not text:
                return "앱 비밀번호를 입력해주세요:"
            self._store_secret_and_ref(["integrations", "gmail", "app_password"], "gmail_app_password", text)
            self._save_integration_field("gmail", "enabled", True)
            self._set_onboarding_stage("wait_integrations")
            return "Gmail이 등록되었습니다!\n\n" + self._integrations_menu()

        # Sub-stages: Google Calendar setup
        if stage == "wait_gcal_creds":
            if not text:
                return "credentials.json 파일 경로를 입력해주세요:"
            path = Path(text.strip()).expanduser()
            if not path.exists():
                return f"파일을 찾을 수 없습니다: {path}\n경로를 다시 입력해주세요:"
            self._save_integration_field("google_calendar", "credentials_json_path", str(path))
            self._save_integration_field("google_calendar", "calendar_id", "primary")
            self._save_integration_field("google_calendar", "enabled", True)
            self._set_onboarding_stage("wait_integrations")
            return (
                "Google Calendar credentials가 등록되었습니다!\n"
                "첫 사용 시 브라우저에서 Google 로그인이 필요합니다.\n\n"
                + self._integrations_menu()
            )

        # Sub-stages: SRT account setup
        if stage == "wait_srt_id":
            if not text:
                return "SRT 멤버십 ID (휴대폰 번호)를 입력해주세요:"
            cfg_path = self.config.workspace / "config.json"
            data = safe_json_load(cfg_path, {})
            data.setdefault("connectors", {})["srt_id"] = text
            write_json(cfg_path, data)
            self._set_onboarding_stage("wait_srt_pw")
            return f"SRT ID: {text}\n비밀번호를 입력해주세요:"

        if stage == "wait_srt_pw":
            if not text:
                return "비밀번호를 입력해주세요:"
            self._store_secret_and_ref(["connectors", "srt_pw"], "srt_pw", text)
            self._set_onboarding_stage("wait_integrations")
            return "SRT 계정이 등록되었습니다!\n\n" + self._integrations_menu()

        # Sub-stages: Telegram bot setup
        if stage == "wait_telegram_token":
            if not text or len(text) < 20 or ":" not in text:
                return "올바른 Telegram Bot 토큰을 입력해주세요 (예: 123456789:ABCdefGHI...):"
            self._store_secret_and_ref(["telegram", "token"], "telegram_token", text)
            self._save_integration_field("telegram", "enabled", True)
            # Also save to telegram section directly
            cfg_path = self.config.workspace / "config.json"
            data = safe_json_load(cfg_path, {})
            data.setdefault("telegram", {})["enabled"] = True
            data["telegram"]["token"] = SecretStore.make_ref("telegram_token")
            data["telegram"].setdefault("allow_from", ["*"])
            write_json(cfg_path, data)
            self._set_onboarding_stage("wait_integrations")
            return (
                "Telegram 봇이 등록되었습니다!\n"
                "실행 시 --mode telegram 또는 --mode all 옵션으로 시작하세요.\n\n"
                + self._integrations_menu()
            )

        # Stage is 'done' but no API key (key was deleted after onboarding?)
        if stage == "done" and not has_primary and not has_fallback:
            if len(text) >= 15 and " " not in text:
                self._store_secret_and_ref(["llm", "api_key"], "llm_api_key", text)
                self.config.api_key = text
                self.client.api_key = text
                return "API 키가 다시 등록되었습니다! 이제 대화를 시작하겠습니다."
            else:
                return "API 키가 설정되어 있지 않습니다. API 키를 입력해주세요."

        return None

    def respond(self, user_text: str, source: str = "cli", user_id: str = "direct") -> str:
        with self._lock:
            # Onboarding check (rule-based, no LLM needed)
            onboarding_response = self._handle_onboarding(user_text)
            if onboarding_response is not None:
                return onboarding_response

            # Pre-check intent to skip web bootstrap for skill tasks
            _, _, pre_prompt_class, _ = self._select_tools_for_query(user_text)

            # Follow-up reclassification: short/ambiguous messages inherit previous context
            _FOLLOWUP_HINTS = (
                "해봐", "해줘", "해", "응", "그래", "진행", "실행", "고", "ㅇㅇ", "넹", "네",
                "좋아", "시작", "부탁", "계속", "다시", "retry", "yes", "go", "ok", "do it",
            )
            if (pre_prompt_class == "analysis_general" or pre_prompt_class == "direct_dialog") and self._last_prompt_class:
                low = user_text.strip().lower()
                is_followup = len(low) <= 10 and any(h in low for h in _FOLLOWUP_HINTS)
                if is_followup:
                    pre_prompt_class = self._last_prompt_class
                    user_text = f"{self._last_user_query} (사용자 추가 지시: {user_text})"
                    self._log_runtime({"type": "followup_reclassify", "original": low, "inherited_class": pre_prompt_class, "merged_query": user_text[:200]})

            fastpath = self._local_fastpath_response(user_text, pre_prompt_class)
            if fastpath is not None:
                stripped = fastpath.strip()
                self.memory.add("user", user_text, {"source": source, "user_id": user_id, "fastpath": True})
                self.memory.add("assistant", stripped, {"source": source, "user_id": user_id, "fastpath": True})
                self._log_runtime({
                    "type": "fastpath_response",
                    "source": source,
                    "user_id": user_id,
                    "prompt_class": pre_prompt_class,
                    "query": user_text[:200],
                })
                self._last_prompt_class = pre_prompt_class
                self._last_user_query = user_text
                return stripped

            skip_web_replan = pre_prompt_class in ("skill_ops", "direct_dialog", "local_time", "memory_focused", "config_ops")
            # Skill/schedule operations should NEVER be decomposed by JobBreaker.
            # The tool (srt_reserve, schedule_task, etc.) handles the full request.
            # JobBreaker decomposition causes confusion: it creates "search first" sub-tasks
            # that waste time on web_search and then the LLM summarizes the empty web results
            # instead of the actual tool output. V4.1 regression fix.
            skip_job_breaker = pre_prompt_class in (
                "skill_ops", "schedule_ops", "workspace_ops",
                "memory_focused", "direct_dialog", "local_time", "email_ops",
            )

            if skip_job_breaker:
                job_plan = {"refined_goal": user_text, "constraints": [], "tasks": [user_text]}
                self._log_runtime({"type": "job_plan_skipped", "pre_prompt_class": pre_prompt_class, "reason": "direct_execution_class"})
            else:
                job_breaker = JobBreakerEngine(self.client)
                job_plan = job_breaker.break_job(user_text)
                self._log_runtime({"type": "job_plan_initial", "plan": job_plan, "pre_prompt_class": pre_prompt_class})

            if not skip_web_replan and not skip_job_breaker and self._plan_needs_web_replan(user_text, job_plan):
                planning_query = ToolDispatcher.normalize_web_search_query(user_text) or user_text.strip()[:240]
                planning_search = self.tools.web_search(planning_query)
                planning_results = planning_search.get("results", []) if isinstance(planning_search, dict) else []
                self._log_runtime({
                    "type": "planning_web_bootstrap", "query": planning_query,
                    "ok": bool(planning_search.get("ok")) if isinstance(planning_search, dict) else False,
                    "result_count": len(planning_results) if isinstance(planning_results, list) else 0,
                    "source": str(planning_search.get("source") or "") if isinstance(planning_search, dict) else "",
                    "error": str(planning_search.get("error") or "") if isinstance(planning_search, dict) else "",
                })
                planning_context = self._summarize_search_for_planning(planning_search)
                if planning_context:
                    job_plan = job_breaker.break_job(user_text, planning_context=planning_context)
                    self._log_runtime({"type": "job_plan_replanned", "plan": job_plan, "reason": "web_bootstrap_context"})

            self._log_runtime({"type": "job_plan", "plan": job_plan})

            messages = self.prompt_builder.build_messages(
                user_text, source=source, user_id=user_id,
                constraints=job_plan.get("constraints") if isinstance(job_plan.get("constraints"), list) else None,
            )
            self.memory.add("user", user_text, {"source": source, "user_id": user_id, "job_plan": job_plan})

            executor = TaskExecutor(self)
            final_text = ""
            tasks = job_plan.get("tasks", [user_text])
            self._log_runtime({"type": "execution_decision", "mode": "single_task_direct" if len(tasks) == 1 else "multi_task_breakdown", "task_count": len(tasks), "tasks": tasks[:8]})

            task_outputs: list[str] = []
            for idx, task in enumerate(tasks):
                if len(tasks) > 1:
                    messages.append({"role": "user", "content": f"[SubTask {idx+1}/{len(tasks)}]: {task}"})
                try:
                    task_output = executor.execute_task(messages, source, user_id, user_original_text=user_text)
                except Exception as e:
                    self._log_runtime({"type": "task_exception_handled", "task_index": idx + 1, "task_total": len(tasks), "error": f"{type(e).__name__}: {safe_str(e)}"})
                    if task_outputs:
                        task_output = "일부 하위 작업에서 일시적 오류가 발생했습니다. 현재까지 수집된 결과를 기준으로 답변을 마무리합니다."
                    else:
                        task_output = "일시적인 LLM/네트워크 오류로 일부 단계 수행에 실패했습니다. 확인 가능한 범위에서 요약하면, 추가 확인이 필요한 상태입니다. 같은 요청을 다시 시도하면 정상 응답될 수 있습니다."
                    task_outputs.append(task_output)
                    break
                task_outputs.append(task_output)
                messages.append({"role": "assistant", "content": task_output})

            if len(tasks) == 1 or len(task_outputs) == 1:
                final_text = task_outputs[0] if task_outputs else ""
            else:
                final_text = self._synthesize_task_outputs(user_text, task_outputs)

            stripped = final_text.strip()
            self.memory.add("assistant", stripped, {"source": source, "user_id": user_id})
            try:
                self._auto_extract_and_store_facts(user_text, stripped, source=source, user_id=user_id)
            except Exception:
                pass
            # Save context for follow-up reclassification
            self._last_prompt_class = pre_prompt_class
            self._last_user_query = user_text
            return stripped


# ═══════════════════════════════════════════════════════════════════════════════
# Section 14-16: Runners (Scheduler, Telegram, CLI)
# ═══════════════════════════════════════════════════════════════════════════════

class SchedulerRunner:
    def __init__(self, engine: AgentEngine, tg_runner: 'TelegramRunner | None'):
        self.engine = engine
        self.tg_runner = tg_runner
        self.stop_event = threading.Event()
        self.task_file = engine.config.workspace / "data" / "scheduled_tasks.jsonl"
        self._last_integration_check: dict[str, float] = {}

    def run_forever(self) -> None:
        print("🕒 [scheduler] background tick started (checking every 30s)")
        while not self.stop_event.is_set():
            if self.task_file.exists():
                self._check_and_run_tasks()
            self._check_integrations()
            time.sleep(30)

    def _check_integrations(self) -> None:
        """Periodically check email/calendar integrations and notify on new content."""
        cfg = safe_json_load(self.engine.config.workspace / "config.json", {})
        integrations = cfg.get("integrations", {})
        now = time.time()
        for provider_key, provider_label in [("gmail", "gmail")]:
            ecfg = integrations.get(provider_key, {})
            if not ecfg.get("enabled"):
                continue
            interval = int(ecfg.get("check_interval_min", 30)) * 60
            last = self._last_integration_check.get(provider_key, 0)
            if now - last < interval:
                continue
            self._last_integration_check[provider_key] = now
            try:
                result = self.engine.tools.check_email(provider=provider_label, limit=5, unread_only=True)
                if result.get("ok") and result.get("count", 0) > 0:
                    lines = [f"📧 {provider_label} 새 메일 {result['count']}건:"]
                    for e in result.get("emails", [])[:3]:
                        lines.append(f"  • {e.get('from', '?')}: {e.get('subject', '(제목 없음)')}")
                    self._notify("\n".join(lines))
            except Exception:
                pass
        gcal = integrations.get("google_calendar", {})
        if gcal.get("enabled"):
            interval = int(gcal.get("check_interval_min", 60)) * 60
            last = self._last_integration_check.get("google_calendar", 0)
            if now - last >= interval:
                self._last_integration_check["google_calendar"] = now
                try:
                    result = self.engine.tools.check_calendar(days_ahead=1, max_results=5)
                    if result.get("ok") and result.get("count", 0) > 0:
                        lines = [f"📅 오늘의 일정 {result['count']}건:"]
                        for ev in result.get("events", [])[:5]:
                            lines.append(f"  • {ev.get('start', '?')} — {ev.get('summary', '(제목 없음)')}")
                        self._notify("\n".join(lines))
                except Exception:
                    pass

    def _notify(self, message: str) -> None:
        if self.tg_runner and self.tg_runner.allow_from:
            for chat_id in self.tg_runner.allow_from:
                if chat_id != "*":
                    self.tg_runner._send_message(chat_id, message)
                    return
        print(f"\n{message}\nyou> ", end="", flush=True)

    @staticmethod
    def _days_in_month(year: int, month: int) -> int:
        if month == 12:
            next_month = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            next_month = datetime(year, month + 1, 1, tzinfo=timezone.utc)
        current = datetime(year, month, 1, tzinfo=timezone.utc)
        return int((next_month - current).days)

    @classmethod
    def _add_months(cls, dt: datetime, months: int) -> datetime:
        months = max(1, int(months))
        year = dt.year + (dt.month - 1 + months) // 12
        month = (dt.month - 1 + months) % 12 + 1
        day = min(dt.day, cls._days_in_month(year, month))
        return dt.replace(year=year, month=month, day=day)

    @classmethod
    def _advance_run_at(cls, current: datetime, recurrence: str, interval: int, now_utc: datetime) -> datetime:
        rec = (recurrence or "none").strip().lower()
        iv = max(1, int(interval or 1))
        next_run = current
        # Catch up if scheduler was down for a long time.
        for _ in range(512):
            if rec == "daily":
                next_run = next_run + timedelta(days=iv)
            elif rec == "weekly":
                next_run = next_run + timedelta(weeks=iv)
            elif rec == "monthly":
                next_run = cls._add_months(next_run, iv)
            else:
                return current
            if next_run > now_utc:
                return next_run
        return next_run

    def _check_and_run_tasks(self) -> None:
        lines = self.task_file.read_text(encoding="utf-8").splitlines()
        pending = []
        now_utc = datetime.now(timezone.utc)
        for raw in lines:
            if not raw.strip():
                continue
            try:
                task = json.loads(raw)
                if task.get("status") != "pending":
                    continue
                run_at = datetime.fromisoformat(task["run_at"])
                if run_at.tzinfo is None:
                    run_at = run_at.replace(tzinfo=timezone.utc)
                else:
                    run_at = run_at.astimezone(timezone.utc)
                recurrence = str(task.get("recurrence") or "none").strip().lower()
                interval = max(1, int(task.get("interval") or 1))
                if now_utc >= run_at:
                    print(f"\n🚀 [scheduler] executing task: {task['prompt']}")
                    internal_prompt = f"[SYSTEM SCHEDULED TASK] Please fulfill this reserved task: {task['prompt']}"
                    try:
                        answer = self.engine.respond(internal_prompt, source="scheduler", user_id="system")
                    except Exception as e:
                        answer = f"Task execution failed: {e}"
                    if self.tg_runner and self.tg_runner.allow_from and "*" not in self.tg_runner.allow_from:
                        target_chat = list(self.tg_runner.allow_from)[0]
                        self.tg_runner._send_message(target_chat, f"⏰ 예약된 작업 결과:\n{answer}")
                    else:
                        print(f"⏰ [예약 작업 결과]\nagent> {answer}\nyou> ", end="", flush=True)
                    if recurrence in {"daily", "weekly", "monthly"}:
                        next_run = self._advance_run_at(run_at, recurrence, interval, now_utc)
                        task["run_at"] = next_run.isoformat()
                        task["last_run_at"] = now_utc.isoformat()
                        task["status"] = "pending"
                        pending.append(task)
                        self.engine._log_runtime({
                            "type": "scheduler_rescheduled",
                            "task_id": str(task.get("id") or ""),
                            "recurrence": recurrence,
                            "interval": interval,
                            "next_run_at": task["run_at"],
                        })
                else:
                    pending.append(task)
            except Exception as e:
                print(f"[scheduler] error parsing task: {e}")
        with self.task_file.open("w", encoding="utf-8") as f:
            for p in pending:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")


class TelegramRunner:
    def __init__(self, engine: AgentEngine, token: str, allow_from: list[str]):
        self.engine = engine
        self.token = token
        self.allow_from = set(allow_from)
        self.offset = 0
        self.stop_event = threading.Event()

    @property
    def _base(self) -> str:
        return f"https://api.telegram.org/bot{self.token}"

    def _allowed(self, chat_id: str) -> bool:
        return "*" in self.allow_from or chat_id in self.allow_from

    def _send_message(self, chat_id: str, text: str) -> None:
        requests.post(f"{self._base}/sendMessage", json={"chat_id": chat_id, "text": text}, timeout=30)

    def run_forever(self) -> None:
        print("📱 [telegram] polling started")
        while not self.stop_event.is_set():
            try:
                resp = requests.get(f"{self._base}/getUpdates", params={"timeout": 30, "offset": self.offset + 1}, timeout=40)
                data = resp.json()
                if not data.get("ok"):
                    time.sleep(2)
                    continue
                for upd in data.get("result", []):
                    self.offset = max(self.offset, int(upd.get("update_id", 0)))
                    msg = upd.get("message") or {}
                    text = msg.get("text")
                    chat = msg.get("chat") or {}
                    chat_id = str(chat.get("id", ""))
                    if not text or not chat_id:
                        continue
                    if not self._allowed(chat_id):
                        continue
                    try:
                        answer = self.engine.respond(text, source="telegram", user_id=chat_id)
                    except Exception as e:
                        answer = f"engine error: {e}"
                    self._send_message(chat_id, answer[:4000])
                    # Check for workspace restart signal
                    if self.engine._pending_restart_agent:
                        self._send_message(chat_id, f"[restart] workspace_{self.engine._pending_restart_agent} 으로 재시작합니다...")
                        self._restart_with_agent(self.engine._pending_restart_agent)
                        return
            except Exception:
                time.sleep(2)

    def _restart_with_agent(self, agent_name: str) -> None:
        """Re-exec the process with the new agent name."""
        import sys, os
        base_dir = str(self.engine.config.workspace.parent)
        args = [sys.executable, sys.argv[0], "--agent", agent_name, "--base-dir", base_dir,
                "--mode", "telegram", "--telegram-token", self.token]
        os.execv(sys.executable, args)

    def stop(self) -> None:
        self.stop_event.set()


class CLIRunner:
    def __init__(self, engine: AgentEngine):
        self.engine = engine

    def run_forever(self) -> None:
        # Trigger onboarding greeting if not yet started
        stage = self.engine._onboarding_stage()
        if not stage:
            greeting = self.engine._handle_onboarding("")
            if greeting:
                print(f"agent> {greeting}")
        elif stage not in ("done",):
            # Resume mid-onboarding (e.g., user restarted CLI)
            prompts = {
                "wait_name": "에이전트 이름을 정해주세요 (영문, 예: fox, nova):",
                "wait_nickname": "닉네임을 지어주세요 (자유, skip 가능):",
                "wait_llm_provider": "LLM 서비스를 선택해주세요 (1~6):",
                "wait_api_key": "API 키를 입력해주세요:",
                "wait_llm_options": "모델 옵션을 선택해주세요 (1: standard, 2: free):",
                "wait_keti_deepseek_key": "DeepSeek API 키를 입력해주세요:",
                "wait_brave_key": "Brave Search API 키를 입력해주세요 (skip 가능):",
                "wait_integrations": "연동할 서비스를 선택해주세요 (gmail/calendar/srt/skip):",
                "wait_gmail_email": "Gmail 주소를 입력해주세요:",
                "wait_gmail_apppass": "Gmail 앱 비밀번호를 입력해주세요:",
                "wait_gcal_creds": "credentials.json 경로를 입력해주세요:",
                "wait_telegram_token": "Telegram Bot 토큰을 입력해주세요:",
                "wait_srt_id": "SRT 멤버십 ID (휴대폰 번호)를 입력해주세요:",
                "wait_srt_pw": "SRT 비밀번호를 입력해주세요:",
            }
            if stage in prompts:
                print(f"agent> {prompts[stage]}")
        else:
            print(f"💻 [cli] interactive mode. type /exit to quit, /reset to reset workspace")
        while True:
            try:
                text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            if text in {"/exit", "exit", "quit", ":q"}:
                break
            if text == "/reset":
                answer = self.engine.tools.reset_workspace("RESET")
                print(f"agent> {json.dumps(answer, ensure_ascii=False)}")
                continue
            try:
                answer = self.engine.respond(text, source="cli", user_id="local")
            except Exception as e:
                answer = f"engine error: {e}"
            print(f"agent> {answer}")
            # Check for workspace restart signal
            if self.engine._pending_restart_agent:
                self._restart_with_agent(self.engine._pending_restart_agent)
                break

    def _restart_with_agent(self, agent_name: str) -> None:
        """Re-exec the process with the new agent name."""
        import sys, os
        base_dir = str(self.engine.config.workspace.parent)
        args = [sys.executable, sys.argv[0], "--agent", agent_name, "--base-dir", base_dir, "--mode", "cli"]
        # Carry over telegram token if present (resolve @secret: refs)
        new_ws = self.engine.config.workspace.parent / f"workspace_{agent_name}"
        cfg = safe_json_load(new_ws / "config.json", {})
        tg_token = cfg.get("telegram", {}).get("token", "")
        if SecretStore.is_ref(tg_token):
            ss = SecretStore(new_ws)
            tg_token = ss.load(SecretStore.ref_key(tg_token)) or ""
        if tg_token:
            args.extend(["--telegram-token", tg_token])
        print(f"[restart] workspace_{agent_name} 으로 재시작합니다...")
        os.execv(sys.executable, args)


class IPCRunner:
    """Machine-readable runner for programmatic interaction (e.g. planaria_tester).

    Protocol (JSON Lines over stdin/stdout):
      Input (one JSON per line):
        {"type":"message","content":"...","source":"tester","user_id":"u1","turn_id":"optional"}
        {"type":"cancel","turn_id":"optional"}
        {"type":"exit"}
      Output (one JSON per line):
        {"type":"ready", ...}  # printed once at startup
        {"type":"tool_call","name":"...","args":{...},"iteration":N,"turn_id":"..."}
        {"type":"reflection_check", ... ,"turn_id":"..."}
        {"type":"reflection_retry", ... ,"turn_id":"..."}
        {"type":"cancel_ack","turn_id":"..."}
        {"type":"response","content":"...","tool_calls":[...],"elapsed_sec":N,"cancelled":bool,"turn_id":"..."}
        {"type":"error","message":"...","turn_id":"..."?}

    All output goes to stdout as single JSON lines (no pretty-print).
    Trace events (tool_call/reflection_*) are emitted live during execution
    via a hook on the engine's runtime log.
    """

    def __init__(self, engine: AgentEngine):
        import sys
        from queue import Queue
        self.engine = engine
        self._sys = sys
        # Save the real stdout for JSON output, then redirect sys.stdout to stderr
        # so that all print() debug output (LLMClient route, scheduler ticks, etc.)
        # goes to stderr instead of polluting the JSON stream.
        self._json_out = sys.stdout
        sys.stdout = sys.stderr
        self._tool_trace: list[dict[str, Any]] = []
        self._current_turn_id: str = ""  # echoed on all messages during a turn
        # Stdin is read on a background thread so the main loop can receive a
        # `cancel` request *while* respond() is still running. Request items
        # land on this queue; the main loop consumes `message`/`exit` events
        # and cancel requests are handled inline by the reader thread (they
        # set a threading.Event on the engine).
        self._request_queue: Queue = Queue()
        self._reader_thread: threading.Thread | None = None
        self._shutdown = False
        self._install_runtime_hook()

    def _emit(self, payload: dict[str, Any]) -> None:
        # Stamp every outbound message with the active turn_id so clients can
        # strictly correlate streamed events with their originating request.
        if self._current_turn_id and "turn_id" not in payload:
            payload = {**payload, "turn_id": self._current_turn_id}
        try:
            self._json_out.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._json_out.flush()
        except Exception:
            pass

    @staticmethod
    def make_ready_event(agent_name: str, onboarding_stage: str, onboarding_required: bool) -> dict[str, Any]:
        stage = (onboarding_stage or "").strip()
        return {
            "type": "ready",
            "agent": agent_name,
            "onboarding_stage": stage or "done",
            "onboarding_required": bool(onboarding_required),
        }

    def _install_runtime_hook(self) -> None:
        """Wrap engine._log_runtime to also emit IPC trace events for tool calls."""
        original_log = self.engine._log_runtime

        def wrapped(payload: dict[str, Any]) -> None:
            original_log(payload)
            try:
                t = payload.get("type", "")
                if t == "tool_call_origin":
                    iteration = payload.get("iteration", 0)
                    details = payload.get("tool_call_details") or []
                    if details:
                        # Preferred path: emit name + args together
                        for d in details:
                            evt = {
                                "type": "tool_call",
                                "name": d.get("name", ""),
                                "args": d.get("args", {}),
                                "iteration": iteration,
                            }
                            self._tool_trace.append(evt)
                            self._emit(evt)
                    else:
                        # Backward-compat fallback: name only
                        for name in payload.get("internal_tools", []) + payload.get("external_tools", []):
                            evt = {"type": "tool_call", "name": name, "args": {}, "iteration": iteration}
                            self._tool_trace.append(evt)
                            self._emit(evt)
                elif t == "reflection_retry":
                    self._emit({
                        "type": "reflection_retry",
                        "prompt_class": payload.get("prompt_class", ""),
                        "correction": payload.get("correction", ""),
                        "iteration": payload.get("iteration", 0),
                    })
                elif t == "response_reflection":
                    self._emit({
                        "type": "reflection_check",
                        "prompt_class": payload.get("prompt_class", ""),
                        "called_tools": payload.get("called_tools", []),
                        "missing_tools": payload.get("missing_tools", []),
                        "verdict": payload.get("reflection_reply", "")[:200],
                    })
            except Exception:
                pass

        self.engine._log_runtime = wrapped  # type: ignore[method-assign]

    def _reader_loop(self) -> None:
        """Background stdin reader: keeps cancel requests responsive while
        the main loop is busy inside engine.respond().

        - `cancel` sets engine._cancel_event and emits cancel_ack immediately.
        - Everything else (message, exit, unknown) is forwarded via queue to
          the main loop for in-order processing.
        """
        while not self._shutdown:
            try:
                line = self._sys.stdin.readline()
            except (EOFError, KeyboardInterrupt):
                self._request_queue.put({"type": "__eof"})
                return
            if not line:
                self._request_queue.put({"type": "__eof"})
                return
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError as e:
                self._emit({"type": "error", "message": f"invalid json: {e}"})
                continue
            rtype = req.get("type", "message")
            if rtype == "cancel":
                # Handle cancel inline (no queueing) so it reaches the engine
                # while respond() is still running.
                self.engine._cancel_event.set()
                self._emit({"type": "cancel_ack", "turn_id": req.get("turn_id", "")})
                continue
            self._request_queue.put(req)

    def run_forever(self) -> None:
        # IPC mode also supports onboarding conversations via normal message exchange.
        stage = self.engine._onboarding_stage()
        has_primary = bool(self.engine.config.api_key)
        has_fallback = bool(self.engine.config.fallback_routes)
        required = not (stage == "done" and (has_primary or has_fallback))
        if not stage and required:
            stage = "wait_name"
        elif stage == "done" and required:
            stage = "wait_api_key"
        self._emit(self.make_ready_event(self.engine.config.agent_name, stage, required))

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        while True:
            req = self._request_queue.get()
            req_type = req.get("type", "message")
            if req_type == "__eof" or req_type == "exit":
                break
            if req_type != "message":
                self._emit({"type": "error", "message": f"unknown request type: {req_type}"})
                continue
            content = str(req.get("content") or "").strip()
            if not content:
                self._emit({"type": "error", "message": "empty message"})
                continue
            source = str(req.get("source") or "ipc")
            user_id = str(req.get("user_id") or "ipc")

            self._current_turn_id = str(req.get("turn_id") or "")
            self._tool_trace = []
            # Clear any stale cancel flag from prior turn
            self.engine._cancel_event.clear()
            t0 = time.time()
            try:
                answer = self.engine.respond(content, source=source, user_id=user_id)
            except Exception as e:
                self._emit({"type": "error", "message": f"engine error: {type(e).__name__}: {e}"})
                self._current_turn_id = ""
                continue
            elapsed = time.time() - t0
            cancelled = self.engine._cancel_event.is_set()
            self._emit({
                "type": "response",
                "content": answer,
                "tool_calls": list(self._tool_trace),
                "elapsed_sec": round(elapsed, 2),
                "cancelled": cancelled,
            })
            self._current_turn_id = ""
            self.engine._cancel_event.clear()

        self._shutdown = True


# ═══════════════════════════════════════════════════════════════════════════════
# Section 18: Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PLANARIA_ONE single-file backend")
    parser.add_argument("--agent", default="default", help="Agent name, workspace_<agent> will be used")
    parser.add_argument("--base-dir", default=".", help="Base directory where workspace_* folders live")
    parser.add_argument("--mode", choices=["cli", "telegram", "all", "ipc"], default="cli", help="Interaction mode (ipc: JSON over stdin/stdout for programmatic clients)")
    parser.add_argument("--init-only", action="store_true", help="Initialize workspace/config and exit")
    parser.add_argument("--message", default="", help="Run one-shot message and exit")
    parser.add_argument("--message-file", default="", help="Read one-shot message from UTF-8 text file")
    parser.add_argument("--source", default="cli", help="Message source label for one-shot mode")
    parser.add_argument("--user-id", default="local", help="User id label for one-shot mode")
    parser.add_argument("--telegram-token", default="", help="Telegram bot token (auto-enables telegram mode)")
    return parser.parse_args()


def _check_runtime_deps() -> None:
    """Verify critical runtime dependencies are importable. Warn loudly on failure."""
    missing: list[str] = []
    try:
        import duckduckgo_search  # noqa: F401
    except ImportError:
        missing.append("duckduckgo-search")
    if missing:
        print("=" * 60)
        print("[startup warning] missing critical dependencies:")
        for m in missing:
            print(f"  - {m}")
        print(f"  python: {sys.executable}")
        print("  fix: launch via './planaria_run.sh' (uses uv venv)")
        print("       or run: uv sync")
        print("=" * 60)


def main() -> int:
    _check_runtime_deps()
    args = parse_args()
    manager = WorkspaceManager(Path(args.base_dir), args.agent)
    manager.ensure()

    if args.init_only:
        print(f"initialized: {manager.root}")
        print(f"config: {manager.config_path}")
        return 0

    cfg = manager.load_config()

    # CLI --telegram-token overrides config and auto-enables telegram
    if args.telegram_token:
        cfg.telegram_token = args.telegram_token
        cfg.telegram_enabled = True
        # Persist to config.json so onboarding doesn't need to re-set
        cfg_data = safe_json_load(manager.config_path, {})
        if "telegram" not in cfg_data:
            cfg_data["telegram"] = {}
        cfg_data["telegram"]["token"] = args.telegram_token
        cfg_data["telegram"]["enabled"] = True
        write_json(manager.config_path, cfg_data)
        # Auto-set mode to telegram if not explicitly "all"
        if args.mode == "cli":
            args.mode = "telegram"

    engine = AgentEngine(cfg)

    one_shot = args.message
    if args.message_file:
        p = Path(args.message_file).expanduser().resolve()
        if not p.exists():
            print(f"[error] message file not found: {p}")
            return 1
        one_shot = p.read_text(encoding="utf-8")

    if one_shot:
        try:
            answer = engine.respond(one_shot, source=args.source, user_id=args.user_id)
        except Exception as e:
            print(f"engine error: {e}")
            return 1
        print(answer)
        return 0

    # IPC mode: minimal — no scheduler, no telegram, no banners. Just JSON I/O.
    if args.mode == "ipc":
        IPCRunner(engine).run_forever()
        return 0

    tg_thread: threading.Thread | None = None
    tg_runner: TelegramRunner | None = None

    if args.mode in {"telegram", "all"}:
        if not cfg.telegram_enabled:
            print("[warn] telegram mode requested but telegram.enabled=false in config.")
            print("       use --telegram-token TOKEN or set telegram.enabled=true in config.json")
        elif not cfg.telegram_token:
            print("[warn] telegram mode requested but telegram.token is empty.")
            print("       use --telegram-token TOKEN or set telegram.token in config.json")
        else:
            tg_runner = TelegramRunner(engine, cfg.telegram_token, cfg.telegram_allow_from)
            tg_thread = threading.Thread(target=tg_runner.run_forever, daemon=True)
            tg_thread.start()

    scheduler = SchedulerRunner(engine, tg_runner)
    scheduler_thread = threading.Thread(target=scheduler.run_forever, daemon=True)
    scheduler_thread.start()

    if args.mode in {"cli", "all"}:
        CLIRunner(engine).run_forever()
    else:
        print("📱 [telegram] running. press Ctrl+C to stop")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    if tg_runner:
        tg_runner.stop()
    scheduler.stop_event.set()
    if tg_thread and tg_thread.is_alive():
        tg_thread.join(timeout=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
