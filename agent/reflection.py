"""
Self-Reflection Mechanism
After each tool call, the agent reflects on whether
it is making progress toward the goal.

Three reflection outcomes:
  CONTINUE   → keep going, on track
  RETRY      → wrong approach, try differently
  FINISH     → task is complete

This is the key sophistication that separates
ToolMind from a basic tool-calling agent.
"""

from __future__ import annotations
import json
from agent.llm import _client
import config

REFLECTION_SYSTEM = """You are a self-reflection module for an AI agent.
Your job is to evaluate whether the agent is making progress
toward its goal and decide what to do next.

Respond ONLY with valid JSON, no markdown:
{
  "progress_assessment": "one sentence on current progress",
  "on_track": true | false,
  "action": "CONTINUE" | "RETRY" | "FINISH",
  "reason": "one sentence explaining your decision",
  "next_suggestion": "what the agent should do next"
}

CONTINUE: agent is making good progress, keep going
RETRY: agent is stuck or going wrong direction, suggest new approach
FINISH: task is fully complete, all questions answered"""


def reflect(
    task: str,
    steps_taken: list,
    last_result: str,
    reflection_num: int,
) -> dict:
    """
    Reflects on the agent's progress.
    Returns a dict with action and suggestions.
    """
    steps_summary = "\n".join(
        f"Step {s['step']}: {s['tool']}({s['args']}) "
        f"→ {s['result'][:100]}..."
        for s in steps_taken[-5:]   # last 5 steps only
    )

    user_content = (
        f"Original task: {task}\n\n"
        f"Steps taken so far:\n{steps_summary}\n\n"
        f"Last tool result:\n{last_result[:300]}\n\n"
        f"This is reflection #{reflection_num}.\n"
        f"Should the agent CONTINUE, RETRY, or FINISH?"
    )

    try:
        response = _client.chat.completions.create(
            model=config.GROQ_MODEL,
            max_tokens=300,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": REFLECTION_SYSTEM
                },
                {"role": "user", "content": user_content},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        raw = raw.strip().strip("```json").strip("```").strip()
        result = json.loads(raw)
        result.setdefault("action", "CONTINUE")
        result.setdefault("next_suggestion", "")
        result.setdefault("on_track", True)
        return result

    except Exception as e:
        return {
            "progress_assessment": f"Reflection failed: {e}",
            "on_track": True,
            "action": "CONTINUE",
            "reason": "Could not reflect",
            "next_suggestion": "",
        }