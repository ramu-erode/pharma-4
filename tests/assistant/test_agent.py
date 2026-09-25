"""The assistant's tool loop, against a scripted stand-in for the Claude API (ADR-0017).

These test the loop's mechanics, not the model's answers.
"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from assistant import agent
from assistant.i3x_client import I3xClient
from tests.i3x import fixtures as fx


def text(t):
    return SimpleNamespace(type="text", text=t)


def tool_use(id_, name, **args):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=args)


def reply(stop, *blocks):
    usage = SimpleNamespace(input_tokens=100, output_tokens=10, cache_read_input_tokens=50)
    return SimpleNamespace(stop_reason=stop, content=list(blocks), usage=usage)


class ScriptedClaude:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.replies.pop(0)


@pytest.fixture
def i3x():
    b = fx.backend()
    b.cache.update(fx.PV_PH, fx.scalar(fx.PV_PH, 7.01, fx.T0))
    return I3xClient("http://testserver/v1", fx.KEY, http=TestClient(fx.app(b)))


def test_tools_run_and_their_results_go_back_in_one_message(i3x):
    llm = ScriptedClaude(
        reply(
            "tool_use",
            tool_use("t1", "read_values", element_ids=[fx.PV_PH]),
            tool_use("t2", "read_values", element_ids=[]),
        ),
        reply("end_turn", text("pH is 7.01.")),
    )
    answer = agent.Assistant(llm, i3x, "claude-opus-5").ask("What is the pH?")
    assert answer.text == "pH is 7.01." and answer.stop == "end_turn"
    assert [c.ok for c in answer.tool_calls] == [True, False]
    results = llm.requests[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert [r["is_error"] for r in results] == [False, True]
    assert "7.01" in results[0]["content"]
    assert answer.usage == {
        "input_tokens": 200,
        "output_tokens": 20,
        "cache_read_input_tokens": 100,
    }


def test_the_question_carries_plant_time_not_the_system_prompt(i3x):
    llm = ScriptedClaude(reply("end_turn", text("ok")))
    agent.Assistant(llm, i3x, "claude-opus-5").ask("Hi")
    first = llm.requests[0]["messages"][0]["content"]
    assert first.startswith("Current plant time: 20") and first.endswith("Hi")
    assert llm.requests[0]["system"] == agent.SYSTEM  # stable, so it caches


def test_request_shape(i3x):
    llm = ScriptedClaude(reply("end_turn", text("ok")))
    agent.Assistant(llm, i3x, "claude-opus-5").ask("Hi")
    req = llm.requests[0]
    assert req["thinking"] == {"type": "adaptive"}
    assert req["fallbacks"] == "default" and req["betas"] == [agent.FALLBACK_BETA]
    other = ScriptedClaude(reply("end_turn", text("ok")))
    agent.Assistant(other, i3x, "claude-sonnet-5").ask("Hi")
    assert "fallbacks" not in other.requests[0]


def test_follow_up_questions_continue_the_conversation(i3x):
    llm = ScriptedClaude(reply("end_turn", text("one")), reply("end_turn", text("two")))
    bot = agent.Assistant(llm, i3x, "claude-opus-5")
    first = bot.ask("Q1")
    second = bot.ask("Q2", first.messages)
    assert second.text == "two" and len(llm.requests[1]["messages"]) == 3


def test_a_refusal_is_reported_and_leaves_history_as_it_was(i3x):
    llm = ScriptedClaude(reply("refusal"))
    answer = agent.Assistant(llm, i3x, "claude-opus-5").ask("Q", [])
    assert answer.stop == "refusal" and answer.text == agent.REFUSED and answer.messages == []


def test_the_loop_is_bounded(i3x):
    loop = [reply("tool_use", tool_use(f"t{i}", "describe_model")) for i in range(3)]
    answer = agent.Assistant(ScriptedClaude(*loop), i3x, "m", max_rounds=3).ask("Q")
    assert answer.stop == "max_rounds" and len(answer.tool_calls) == 3


def test_a_truncated_answer_says_so(i3x):
    answer = agent.Assistant(ScriptedClaude(reply("max_tokens", text("partial"))), i3x, "m").ask(
        "Q"
    )
    assert answer.text.startswith("partial") and "cut off" in answer.text


def test_no_key_means_unavailable_not_a_crash():
    settings = SimpleNamespace(anthropic_api_key=None)
    with pytest.raises(agent.AssistantUnavailable, match="ANTHROPIC_API_KEY"):
        agent.from_settings(settings)
