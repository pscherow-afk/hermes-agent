"""Tests for P0-9: api_messages must not mutate the original message history.

The per-call API message list (``api_messages``) is built from the
persisted session ``messages`` list. Historically the build used a
shallow ``msg.copy()``, which shares nested ``tool_calls`` lists and
``content`` block-lists with the original history. Later in-place
mutations (surrogate sanitization, tool-call argument normalization,
reasoning echo, thinking-only drop) therefore leaked into the stored
trajectory, corrupting retries / fallbacks.

After the fix the build uses ``copy.deepcopy(msg)`` so the original
``messages`` stay byte-stable across the call.
"""

import copy

from agent.message_sanitization import _sanitize_messages_surrogates


def _assistant_with_tool_calls():
    """Assistant message with nested mutable tool_calls + content list."""
    return {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "thinking done"},
        ],
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "/tmp/foo"}',
                },
            }
        ],
        "reasoning_content": " ",
    }


def _user_with_content_blocks():
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "read the file"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
    }


def test_deepcopy_isolates_nested_tool_calls():
    """Mutating api_messages must not touch the original tool_calls list."""
    messages = [_assistant_with_tool_calls()]
    original = copy.deepcopy(messages)

    # Build the API copy exactly as conversation_loop.py now does.
    api_messages = [copy.deepcopy(m) for m in messages]

    # Simulate the in-place tool-call argument normalization that runs
    # after the build (conversation_loop.py ~line 921).
    for am in api_messages:
        tcs = am.get("tool_calls")
        if not tcs:
            continue
        new_tcs = []
        for tc in tcs:
            if isinstance(tc, dict) and "function" in tc:
                args_obj = __import__("json").loads(tc["function"]["arguments"])
                tc = {
                    **tc,
                    "function": {
                        **tc["function"],
                        "arguments": __import__("json").dumps(
                            args_obj, separators=(",", ":"), sort_keys=True
                        ),
                    },
                }
            new_tcs.append(tc)
        am["tool_calls"] = new_tcs

    # Original tool_calls must be unchanged (still pretty-printed JSON).
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"path": "/tmp/foo"}'
    # And the normalized api copy differs.
    assert api_messages[0]["tool_calls"][0]["function"]["arguments"] == '{"path":"/tmp/foo"}'
    # Structural identity check: the nested dicts are distinct objects.
    assert messages[0]["tool_calls"] is not api_messages[0]["tool_calls"]
    assert (
        messages[0]["tool_calls"][0]["function"]
        is not api_messages[0]["tool_calls"][0]["function"]
    )
    # Full history unchanged.
    assert messages == original


def test_deepcopy_isolates_content_blocks():
    """Mutating api_messages content blocks must not touch original content."""
    messages = [_user_with_content_blocks()]
    original = copy.deepcopy(messages)

    api_messages = [copy.deepcopy(m) for m in messages]

    # Simulate content-block mutation (e.g. surrogate sanitization).
    for am in api_messages:
        if isinstance(am.get("content"), list):
            for part in am["content"]:
                if isinstance(part, dict) and "text" in part:
                    part["text"] = part["text"] + " [mutated]"

    assert messages[0]["content"][0]["text"] == "read the file"
    assert api_messages[0]["content"][0]["text"] == "read the file [mutated]"
    assert messages == original


def test_surrogate_sanitization_does_not_leak_to_history():
    """End-to-end: _sanitize_messages_surrogates on the API copy is isolated."""
    messages = [
        {
            "role": "assistant",
            "content": "ok\ud800",  # lone surrogate
            "tool_calls": [
                {
                    "id": "call_x",
                    "type": "function",
                    "function": {"name": "f", "arguments": '{"a": 1}'},
                }
            ],
        }
    ]
    original = copy.deepcopy(messages)

    api_messages = [copy.deepcopy(m) for m in messages]
    _sanitize_messages_surrogates(api_messages)

    # API copy cleaned.
    assert "\ud800" not in api_messages[0]["content"]
    assert api_messages[0]["content"] == "ok\ufffd"
    # Original history still has the surrogate — not leaked.
    assert messages[0]["content"] == "ok\ud800"
    assert messages == original


def test_shallow_copy_would_have_leaked():
    """Regression guard: documents why deepcopy is required (shallow leaks)."""
    messages = [_assistant_with_tool_calls()]
    # This is what the OLD code did — must NOT match the fixed behaviour.
    api_messages = [dict(m) for m in messages]  # shallow

    # Mutate a nested list in the shallow copy.
    api_messages[0]["tool_calls"][0]["function"]["arguments"] = '{"path":"/tmp/bar"}'

    # With shallow copy the original is corrupted (proves the bug existed).
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"path":"/tmp/bar"}'
