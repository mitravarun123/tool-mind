"""
Groq LLM client for ToolMind agent.
"""

from __future__ import annotations
import os
import json
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
import config

load_dotenv()

_client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

AGENT_SYSTEM = """You are ToolMind, an expert data analysis agent.
You solve tasks by calling tools one step at a time.

Available tools:
- load_csv(filepath)              load a CSV file
- describe_data()                 get data overview
- run_stats(column, group_by)     compute statistics
- filter_data(condition)          filter rows
- plot_chart(chart_type, x, y, title)  generate a chart
- pair-plot(chart_type,attribute,title) generate a pair plot
- run_python(code)                execute Python code
- search_web(query)               search the web
- write_report(title, content)    save findings to markdown
- finish(answer)                  return final answer

ALWAYS respond with a JSON object:
{
  "thought": "your reasoning about what to do next",
  "tool": "tool_name",
  "args": {"arg1": "value1", "arg2": "value2"}
}

Rules:
1. Think step by step
2. Use describe_data before running stats
3. Always plot a chart to visualize findings
4. Always write_report before finish
5. If a tool fails, try a different approach
6. Never call the same tool with same args twice"""


def call_agent(
    messages: List[Dict],
    system: str = None,
) -> str:
    """Single LLM call returning raw text."""
    system = system or AGENT_SYSTEM
    response = _client.chat.completions.create(
        model=config.GROQ_MODEL,
        max_tokens=config.MAX_TOKENS,
        temperature=config.TEMPERATURE,
        messages=[
            {"role": "system", "content": system}
        ] + messages,
    )
    return response.choices[0].message.content or ""


def parse_tool_call(raw: str) -> Dict:
    """
    Parses LLM output into a tool call dict.
    Handles cases where LLM adds extra text around JSON.
    """
    try:
        raw = raw.strip()
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]
        result = json.loads(raw)
        result.setdefault("thought", "")
        result.setdefault("tool", "finish")
        result.setdefault("args", {})
        return result
    except Exception:
        return {
            "thought": "Failed to parse response",
            "tool": "finish",
            "args": {"answer": raw},
        }