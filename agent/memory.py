"""
Agent Memory
Tracks the full history of thoughts, tool calls, and results.
Provides context to the agent for each new step.
Prevents the agent from repeating failed attempts.
"""

from __future__ import annotations
from typing import List, Dict, Optional


class AgentMemory:
    """
    Stores the agent's working memory for one task.
    Acts as the conversation history fed back to the LLM.
    """

    def __init__(self, task: str):
        self.task = task
        self.steps: List[Dict] = []
        self.tool_call_counts: Dict[str, int] = {}
        self.failed_attempts: List[str] = []
        self.final_answer: Optional[str] = None

    def add_step(
        self,
        thought: str,
        tool: str,
        args: dict,
        result: str,
    ):
        step = {
            "step": len(self.steps) + 1,
            "thought": thought,
            "tool": tool,
            "args": args,
            "result": result,
        }
        self.steps.append(step)
        self.tool_call_counts[tool] = (
            self.tool_call_counts.get(tool, 0) + 1
        )

        if result.startswith("❌"):
            self.failed_attempts.append(
                f"Step {step['step']}: {tool}({args}) → {result}"
            )

    def already_tried(self, tool: str, args: dict) -> bool:
        """Checks if this exact tool+args combo was already tried."""
        for step in self.steps:
            if step["tool"] == tool and step["args"] == args:
                return True
        return False

    def to_messages(self) -> List[Dict]:
        """
        Converts memory to LLM message format.
        Each step becomes a user+assistant exchange.
        """
        messages = [
            {
                "role": "user",
                "content": f"Task: {self.task}"
            }
        ]

        for step in self.steps:
            # Assistant's tool call
            messages.append({
                "role": "assistant",
                "content": (
                    f'{{"thought": "{step["thought"]}", '
                    f'"tool": "{step["tool"]}", '
                    f'"args": {step["args"]}}}'
                ),
            })
            # Tool result as user message
            messages.append({
                "role": "user",
                "content": (
                    f"Tool result (Step {step['step']}):\n"
                    f"{step['result']}\n\n"
                    f"Continue with the next step."
                ),
            })

        return messages

    def summary(self) -> str:
        """Human-readable summary of what the agent did."""
        lines = [
            f"Task: {self.task}",
            f"Steps taken: {len(self.steps)}",
            f"Tools used: {dict(self.tool_call_counts)}",
        ]
        if self.failed_attempts:
            lines.append(
                f"Failed attempts: {len(self.failed_attempts)}"
            )
        if self.final_answer:
            lines.append(f"Answer: {self.final_answer[:100]}...")
        return "\n".join(lines)