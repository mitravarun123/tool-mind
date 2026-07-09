"""
Agent Executor
The main loop that runs the agent:
  Think → Call Tool → Get Result → Reflect → Repeat

Stops when:
  - Agent calls finish()
  - Max steps reached
  - Reflection says FINISH
"""

from __future__ import annotations
import os
from typing import Optional

from agent.llm import call_agent, parse_tool_call
from agent.tools import execute_tool
from agent.memory import AgentMemory
from agent.reflection import reflect
import config


def run_agent(task: str) -> str:
    """
    Runs the full agent loop for a given task.
    Returns the final answer.
    """
    print(f"\n{'='*60}")
    print(f"  ToolMind Agent")
    print(f"  Task: {task}")
    print(f"{'='*60}")

    memory = AgentMemory(task)
    reflection_count = 0
    final_answer = ""

    for step_num in range(1, config.MAX_STEPS + 1):
        print(f"\n[Step {step_num}/{config.MAX_STEPS}]")

        # Get LLM decision
        messages = memory.to_messages()
        raw_response = call_agent(messages)
        tool_call = parse_tool_call(raw_response)

        thought = tool_call.get("thought", "")
        tool    = tool_call.get("tool", "finish")
        args    = tool_call.get("args", {})

        print(f"  Thought: {thought[:80]}...")
        print(f"  Tool: {tool}({args})")

        # Check if already tried this exact call
        if memory.already_tried(tool, args) and tool != "finish":
            print(f"  ⚠️  Already tried {tool}({args}) — skipping")
            # Add a note to memory and continue
            memory.add_step(
                thought=thought,
                tool=tool,
                args=args,
                result=(
                    "⚠️ Already tried this exact call. "
                    "Try a different approach."
                ),
            )
            continue

        # Handle finish
        if tool == "finish":
            final_answer = args.get("answer", "Task complete.")
            memory.final_answer = final_answer
            print(f"  ✅ Agent finished")
            break

        # Execute tool
        result = execute_tool(tool, args)
        print(f"  Result: {result[:100]}...")

        # Store in memory
        memory.add_step(
            thought=thought,
            tool=tool,
            args=args,
            result=result,
        )

        # Self-reflection every 3 steps
        if step_num % 3 == 0:
            reflection_count += 1
            print(f"\n  [Reflection {reflection_count}]")

            reflection = reflect(
                task=task,
                steps_taken=memory.steps,
                last_result=result,
                reflection_num=reflection_count,
            )

            print(f"  Assessment: {reflection['progress_assessment']}")
            print(f"  Action: {reflection['action']}")

            if reflection["action"] == "FINISH":
                print(f"  Reflection says task is complete")
                # One more step to write report and finish
                messages = memory.to_messages()
                messages.append({
                    "role": "user",
                    "content": (
                        "The reflection module says the task is "
                        "complete. Please write_report with your "
                        "findings, then call finish with your "
                        "final answer."
                    ),
                })
                raw = call_agent(messages)
                tool_call = parse_tool_call(raw)

                if tool_call["tool"] == "write_report":
                    r = execute_tool(
                        "write_report", tool_call["args"]
                    )
                    memory.add_step(
                        thought="Writing final report",
                        tool="write_report",
                        args=tool_call["args"],
                        result=r,
                    )

                final_answer = reflection.get(
                    "progress_assessment", "Task complete."
                )
                memory.final_answer = final_answer
                break

            elif reflection["action"] == "RETRY":
                suggestion = reflection.get(
                    "next_suggestion", ""
                )
                if suggestion:
                    print(f"  Suggestion: {suggestion}")
                    # Add suggestion to memory
                    memory.add_step(
                        thought="Reflection suggested retry",
                        tool="reflect",
                        args={"suggestion": suggestion},
                        result=(
                            f"Retry suggested: {suggestion}"
                        ),
                    )

            if reflection_count >= config.MAX_REFLECTION_DEPTH:
                print(f"  Max reflections reached")
                break

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Agent Summary")
    print(f"  {memory.summary()}")
    print(f"{'='*60}")

    return final_answer or "Agent completed without explicit answer."