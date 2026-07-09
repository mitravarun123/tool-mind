"""
ToolMind config.
"""

# ── LLM ──────────────────────────────────────────────────────
GROQ_MODEL    = "llama-3.3-70b-versatile"
MAX_TOKENS    = 1000
TEMPERATURE   = 0.2

# ── Agent Loop ────────────────────────────────────────────────
MAX_STEPS           = 15   # max tool calls per task
MAX_RETRIES         = 3    # max self-reflection retries
MAX_REFLECTION_DEPTH = 3   # how deep reflection goes

# ── Tools ─────────────────────────────────────────────────────
AVAILABLE_TOOLS = [
    "load_csv",
    "describe_data",
    "run_stats",
    "filter_data",
    "plot_chart",
    "run_python",
    "search_web",
    "write_report",
    "finish",
]

# ── Output ────────────────────────────────────────────────────
OUTPUT_DIR  = "outputs"
CHARTS_DIR  = "outputs/charts"
REPORTS_DIR = "outputs/reports"
DATA_DIR    = "data"