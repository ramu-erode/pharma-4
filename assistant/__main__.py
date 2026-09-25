"""Ask the plant a question from the command line (ADR-0017).

    python -m assistant "Why was B2026-0142's titer low?"

Needs ANTHROPIC_API_KEY in .env and the i3x service running.
"""

from __future__ import annotations

import sys

import anthropic

from assistant.agent import AssistantUnavailable, from_settings, trail
from common.settings import get_settings


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if argv else 2
    try:
        assistant = from_settings(get_settings())
        answer = assistant.ask(" ".join(argv))
    except AssistantUnavailable as exc:
        print(exc, file=sys.stderr)
        return 1
    except anthropic.APIError as exc:
        print(f"Claude API error: {exc}", file=sys.stderr)
        return 1
    if answer.tool_calls:
        print(trail(answer.tool_calls), file=sys.stderr)
        print(file=sys.stderr)
    print(answer.text)
    u = answer.usage
    print(
        f"\n[{answer.stop}; {len(answer.tool_calls)} tool calls; "
        f"{u.get('input_tokens', 0)} in / {u.get('output_tokens', 0)} out / "
        f"{u.get('cache_read_input_tokens', 0)} cached tokens]",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
