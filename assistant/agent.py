"""The assistant: Claude answering plant questions through i3X only (ADR-0017).

A hand-written tool loop, bounded by `max_rounds`. The conversation (`Answer.messages`)
is returned so a UI can ask follow-up questions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import anthropic

from assistant import tools
from assistant.i3x_client import I3xClient, I3xError
from common.settings import Settings

# Server-side refusal fallbacks ("default" routing) for the models that support them.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = {"claude-opus-5"}

SYSTEM = """\
You answer questions about pharma-4, a simulated biopharmaceutical plant: two 2000 L
fed-batch CHO bioreactors (BR-101, BR-102) making a monoclonal antibody. You see the
plant only through an i3X 1.0 API, which serves the plant model, batch context, live
values and history. Your tools map one to one onto it.

The process. A batch runs about 14 days: growth, a temperature shift around day 5, then
production and harvest. Titer (g/L) at harvest is the yield. Five levers drive titer:
shift_day, prod_temp (°C after the shift), ph_sp, do_sp (% air saturation) and
feed_mult. Each recipe sets their nominal values and proven acceptable ranges (PARs).
Recipes v1 and v2 are legacy and v3 is current. MFG batches are manufacturing; PC
batches are the process-characterisation DoE across the PARs. Operations (Setup,
Inoculation, Growth, TempShift, Production, Harvest) run in sequence. Phases (TEMP_CTRL,
PH_CTRL, DO_CTRL, FEED_ADD) run in parallel, and each one controls or only monitors
certain equipment modules (Controls / Monitors relationships).

The address space. Browse roots: `pharmaco` (site > area > line > bioreactor, and inside
each bioreactor its equipment modules, control modules, sensors and data points),
`batches`, `recipes` and `phase-classes`. A data point's elementId is its UNS topic, e.g.
pharmaco/chennai/upstream/suite-1/BR-101/pv/ph (pv = process value, sp = setpoint,
lab = daily offline sample, state = batch/operation/phase, ai = model outputs). A batch
(elementId like B2026-0142) has a value with its recipe, planned and actual levers,
outcome and operation/phase timeline. Its children are its anomaly alerts, operator
actions and yield recommendations. Use the batch's start and end as the history window
for its data points.

AI outputs in the data are advisory: anomaly alerts (rules, stats, mspc, iforest layers;
score against threshold), titer predictions (P10/P50/P90) and setpoint recommendations.

Time. Every timestamp is simulated plant time in UTC. It can be years away from today's
date, so never compare it with the real clock. The user's message states the current
plant time.

Answering:
- Look before you answer. Name the elementIds and time windows your answer rests on.
- Keep what the data shows separate from what you infer from it. Say how confident you
  are, and what reading would confirm or rule out your explanation.
- For "why" questions about one batch, compare it with its peers, such as batches on
  the same recipe and campaign.
- You are decision support. You cannot change the plant, and you do not make batch
  disposition or release decisions. If asked to, say that a person decides through the
  approved process.
- If the data cannot answer the question, say so plainly.
- Lead with the answer, then the evidence, with numbers and units.
"""

REFUSED = (
    "The model declined to answer this question. Rephrase it around the plant data, or "
    "ask a person."
)


class AssistantUnavailable(RuntimeError):
    """No API key, or no i3X server: the Ask page and CLI explain which."""


@dataclass
class ToolCall:
    name: str
    input: dict[str, Any]
    ok: bool
    detail: str  # the error, or the size of the result


@dataclass
class Answer:
    text: str
    stop: str  # end_turn | max_tokens | refusal | max_rounds
    tool_calls: list[ToolCall]
    messages: list[dict[str, Any]]
    usage: dict[str, int] = field(default_factory=dict)


class Assistant:
    def __init__(
        self,
        llm: Any,  # anthropic.Anthropic, or a stand-in with .beta.messages.create
        i3x: I3xClient,
        model: str,
        max_rounds: int = 16,
        max_tokens: int = 16_000,
    ) -> None:
        self.llm = llm
        self.i3x = i3x
        self.model = model
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens

    def plant_time(self) -> str:
        """The plant-time high-water mark: the timestamp of a static object's value."""
        [result] = self.i3x.value(["pharmaco"])
        return result["result"]["timestamp"]

    def ask(self, question: str, history: list[dict[str, Any]] | None = None) -> Answer:
        messages = list(history or [])
        before = len(messages)
        messages.append(
            {"role": "user", "content": f"Current plant time: {self.plant_time()}\n\n{question}"}
        )
        calls: list[ToolCall] = []
        usage: dict[str, int] = {}
        for _ in range(self.max_rounds):
            response = self._create(messages)
            for k in ("input_tokens", "output_tokens", "cache_read_input_tokens"):
                usage[k] = usage.get(k, 0) + (getattr(response.usage, k, 0) or 0)
            if response.stop_reason == "refusal":
                return Answer(REFUSED, "refusal", calls, messages[:before], usage)
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason == "pause_turn":
                continue
            if response.stop_reason != "tool_use":
                return Answer(_text(response), response.stop_reason, calls, messages, usage)
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result, call = self._run_tool(block.name, block.input)
                calls.append(call)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                        "is_error": not call.ok,
                    }
                )
            messages.append({"role": "user", "content": results})
        return Answer(
            f"Stopped after {self.max_rounds} tool rounds without a final answer. Ask a "
            "narrower question.",
            "max_rounds",
            calls,
            messages,
            usage,
        )

    def _create(self, messages: list[dict[str, Any]]) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM,
            "tools": tools.TOOLS,
            "messages": messages,
            "thinking": {"type": "adaptive"},
            "cache_control": {"type": "ephemeral"},  # the growing conversation, each round
        }
        if self.model in FALLBACK_MODELS:
            kwargs |= {"betas": [FALLBACK_BETA], "fallbacks": "default"}
        return self.llm.beta.messages.create(**kwargs)

    def _run_tool(self, name: str, args: Any) -> tuple[str, ToolCall]:
        args = args if isinstance(args, dict) else {}
        try:
            result = tools.run(self.i3x, name, args)
        except tools.ToolError as exc:
            return f"Error: {exc}", ToolCall(name, args, False, str(exc))
        return result, ToolCall(name, args, True, f"{len(result)} characters")


def _text(response: Any) -> str:
    text = "\n\n".join(b.text for b in response.content if b.type == "text").strip()
    if response.stop_reason == "max_tokens":
        text += "\n\n[The answer was cut off at the output limit.]"
    return text or "(no answer)"


def from_settings(settings: Settings) -> Assistant:
    """An assistant wired to the configured model and i3X server."""
    if not settings.anthropic_api_key:
        raise AssistantUnavailable(
            "Set ANTHROPIC_API_KEY in .env (never in .env.example), then restart the dashboard."
        )
    i3x = I3xClient(settings.i3x_url, settings.i3x_api_key)
    try:
        i3x.info()
    except I3xError as exc:
        raise AssistantUnavailable(f"The i3X server is not answering: {exc}") from exc
    llm = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return Assistant(llm, i3x, settings.assistant_model)


def trail(calls: list[ToolCall]) -> str:
    """One line per tool call, for the CLI and the Ask page."""
    return "\n".join(
        f"{'✓' if c.ok else '✗'} {c.name}({json.dumps(c.input, ensure_ascii=False)}) — {c.detail}"
        for c in calls
    )
