"""Provider-specific request controls for model replay runners.

The benchmark protocol remains model-agnostic.  This tiny adapter only adds
the provider field needed to explicitly disable hidden reasoning; it never
changes prompts, skills, evaluators, or token caps.
"""

from __future__ import annotations


def reasoning_extra_body(model: str, *, thinking: bool = False) -> dict | None:
    """Return an explicit reasoning control for providers that expose one."""

    basename = model.lower().rsplit("/", 1)[-1]
    if "qwen3" in basename:
        return {"chat_template_kwargs": {"enable_thinking": thinking}}
    if "glm-5" in basename or "kimi" in basename:
        return {"enable_thinking": thinking}
    if "gpt-5" in basename:
        return {"reasoning_effort": "minimal"}
    return None
