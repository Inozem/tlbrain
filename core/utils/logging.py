import json
import logging
import re

_STANDARD_FIELDS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "taskName",
})

_SENSITIVE_FIELDS = frozenset({
    "token", "accesstoken", "refreshtoken", "idtoken", "clientsecret",
    "authorization", "credential", "credentials", "password", "passwd",
    "secret", "state", "code", "authorizationcode", "proxyerror",
    "request", "response", "requestbody", "requestheaders", "responsebody",
    "responseheaders",
})
_SENSITIVE_TEXT = re.compile(
    r"\b[\w.+/-]*(?:access[_-]?token|refresh[_-]?token|id[_-]?token|"
    r"client[_-]?secret|credential|authorization)[\w.+/-]*\b(?![\"']\s*:)" ,
    re.IGNORECASE,
)
_BEARER_TEXT = re.compile(r"\bBearer\s+[^\s,;\]}'\"]+", re.IGNORECASE)
_QUERY_VALUE = re.compile(r"([?&][^=&#\s]+=)[^&#\s]*")
_QUOTED_ASSIGNED_SECRET = re.compile(
    r"([\"'])(access[_-]?token|refresh[_-]?token|id[_-]?token|client[_-]?secret|"
    r"authorization|credential|password|passwd|secret|state|code)\1"
    r"(\s*[:=]\s*)([\"'])(.*?)(?<!\\)\4",
    re.IGNORECASE,
)
_ASSIGNED_SECRET = re.compile(
    r"(\b(?:access[_-]?token|refresh[_-]?token|id[_-]?token|client[_-]?secret|"
    r"authorization|credential|password|passwd|secret|state|code)\b\s*[:=]\s*)"
    r"([^\s&,;\]}\"']+)",
    re.IGNORECASE,
)


def _is_sensitive_field(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", key).lower()
    if normalized in {"tokencount", "refreshtokencount", "accesstokencount", "idtokencount"}:
        return False
    if normalized in _SENSITIVE_FIELDS:
        return True
    if "token" in normalized or "proxyerror" in normalized:
        return True
    if any(term in normalized for term in (
        "clientsecret", "secret", "credential", "password", "passwd", "authorization"
    )):
        return True
    if normalized.startswith(("oauthstate", "authstate")):
        return True
    if any(normalized.startswith(term) for term in (
        "requestbody", "requestheaders", "responsebody", "responseheaders"
    )):
        return True
    return False


def _sanitize_text(value: str) -> str:
    value = _QUERY_VALUE.sub(r"\1[REDACTED]", value)
    value = _BEARER_TEXT.sub("Bearer [REDACTED]", value)
    value = _QUOTED_ASSIGNED_SECRET.sub(r"\1\2\1\3\4[REDACTED]\4", value)
    value = _ASSIGNED_SECRET.sub(r"\1[REDACTED]", value)
    return _SENSITIVE_TEXT.sub("[REDACTED]", value)


def _sanitize_extra(value):
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if _is_sensitive_field(str(key)):
                continue
            safe_key = _sanitize_text(str(key))
            sanitized[safe_key] = _sanitize_extra(item)
        return sanitized
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_extra(item) for item in value]
    if isinstance(value, str):
        return _sanitize_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _sanitize_text(str(value))


class _CloudRunFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = _sanitize_text(record.getMessage())
        log: dict = {
            "severity": record.levelname,
            "message": message,
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_FIELDS:
                if _is_sensitive_field(key):
                    continue
                log[key] = _sanitize_extra(value)
        if record.exc_info:
            log["exception"] = _sanitize_text(self.formatException(record.exc_info))
        return json.dumps(log, ensure_ascii=False, default=str)


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_CloudRunFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
