"""
All 8 tools available to the ToolMind agent.
Each tool is a plain Python function that returns
a string result (which goes back into the agent's context).
"""

from __future__ import annotations
import os
import io
import sys
import traceback
import json
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
import config

# ── Shared dataframe state ────────────────────────────────────
# The agent works with one dataframe at a time
_current_df: Optional[pd.DataFrame] = None
_current_filepath: str = ""


def load_csv(filepath: str) -> str:
    """Loads a CSV file into memory."""
    global _current_df, _current_filepath

    # Try relative and absolute paths
    paths_to_try = [
        filepath,
        os.path.join(config.DATA_DIR, filepath),
        os.path.join(config.DATA_DIR,
                     os.path.basename(filepath)),
    ]

    for path in paths_to_try:
        if os.path.exists(path):
            try:
                _current_df = pd.read_csv(path)
                _current_filepath = path
                return (
                    f"✅ Loaded '{path}' successfully.\n"
                    f"Shape: {_current_df.shape[0]} rows × "
                    f"{_current_df.shape[1]} columns\n"
                    f"Columns: {list(_current_df.columns)}"
                )
            except Exception as e:
                return f"❌ Error loading CSV: {e}"

    return (
        f"❌ File not found: {filepath}\n"
        f"Available files in data/: "
        f"{os.listdir(config.DATA_DIR) if os.path.exists(config.DATA_DIR) else 'directory not found'}"
    )


def describe_data() -> str:
    """Returns a comprehensive overview of the loaded data."""
    if _current_df is None:
        return "❌ No data loaded. Call load_csv first."

    df = _current_df
    lines = [
        f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
        f"\nColumn types:",
    ]
    for col in df.columns:
        dtype = str(df[col].dtype)
        nulls = df[col].isnull().sum()
        unique = df[col].nunique()
        lines.append(
            f"  {col}: {dtype} | "
            f"{nulls} nulls | {unique} unique values"
        )

    lines.append(f"\nFirst 3 rows:")
    lines.append(df.head(3).to_string())

    lines.append(f"\nNumeric summary:")
    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        lines.append(numeric.describe().round(2).to_string())

    return "\n".join(lines)


def run_stats(
    column: str = None,
    group_by: str = None,
) -> str:
    if _current_df is None:
        return "❌ No data loaded. Call load_csv first."

    df = _current_df
    results = []

    try:
        # Handle comma-separated group_by string
        # e.g. "region,product" → ["region", "product"]
        if group_by:
            if isinstance(group_by, list):
                group_cols = [
                    c.strip() for c in group_by
                    if c.strip() in df.columns
                ]
            else:
                group_cols = [
                    c.strip() for c in group_by.split(",")
                    if c.strip() in df.columns
                ]
        else:
            group_cols = []

        if group_cols:
            if column and column in df.columns:
                stats = df.groupby(group_cols)[column].agg([
                    "mean", "sum", "min", "max", "count"
                ]).round(2)
                results.append(
                    f"Stats for '{column}' grouped by "
                    f"{group_cols}:\n{stats.to_string()}"
                )
            else:
                counts = df.groupby(group_cols).size(
                ).reset_index(name="count")
                results.append(
                    f"Counts grouped by {group_cols}:\n"
                    f"{counts.to_string()}"
                )

        elif column and column in df.columns:
            s = df[column]
            results.append(
                f"Stats for '{column}':\n"
                f"  Mean:   {s.mean():.2f}\n"
                f"  Median: {s.median():.2f}\n"
                f"  Std:    {s.std():.2f}\n"
                f"  Min:    {s.min():.2f}\n"
                f"  Max:    {s.max():.2f}\n"
                f"  Sum:    {s.sum():.2f}"
            )
        else:
            numeric = df.select_dtypes(include=[np.number])
            if not numeric.empty:
                corr = numeric.corr().round(3)
                results.append(
                    f"Correlation matrix:\n{corr.to_string()}"
                )

        return "\n\n".join(results) if results \
               else "No numeric columns found."

    except Exception as e:
        return f"❌ Stats error: {e}"

def filter_data(condition: str) -> str:
    """
    Filters the dataframe by a condition string.
    Example: "Sales > 1000" or "Region == 'North'"
    """
    global _current_df

    if _current_df is None:
        return "❌ No data loaded. Call load_csv first."

    try:
        original_len = len(_current_df)
        filtered = _current_df.query(condition)
        _current_df = filtered

        return (
            f"✅ Filtered by: {condition}\n"
            f"Rows before: {original_len}\n"
            f"Rows after:  {len(filtered)}\n"
            f"Removed:     {original_len - len(filtered)}"
        )
    except Exception as e:
        return f"❌ Filter error: {e}\nTry: column_name > value"


    
def plot_chart(
    chart_type: str = "bar",
    x: str = None,
    y: str = None,
    title: str = "Chart",
    hue: str = None,
) -> str:
    """
    Generates a matplotlib chart and saves it to outputs/charts/.
    chart_type: bar, line, scatter, hist, box, heatmap
    """
    if _current_df is None:
        return "❌ No data loaded. Call load_csv first."

    os.makedirs(config.CHARTS_DIR, exist_ok=True)

    df = _current_df
    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    try:
        if chart_type == "bar":
            if x and y and x in df.columns and y in df.columns:
                plot_data = df.groupby(x)[y].mean().reset_index()
                sns.barplot(data=plot_data, x=x, y=y,
                            ax=ax, palette="Blues_d")
            else:
                df.select_dtypes(include=[np.number]).mean(
                ).plot(kind="bar", ax=ax, color="#2196F3")

        elif chart_type == "line":
            if x and y and x in df.columns and y in df.columns:
                sns.lineplot(data=df, x=x, y=y,
                             hue=hue, ax=ax)
            else:
                df.select_dtypes(
                    include=[np.number]
                ).plot(ax=ax)

        elif chart_type == "scatter":
            if x and y and x in df.columns and y in df.columns:
                sns.scatterplot(
                    data=df, x=x, y=y,
                    hue=hue, ax=ax
                )

        elif chart_type == "hist":
            col = y or x
            if col and col in df.columns:
                df[col].hist(ax=ax, bins=20,
                             color="#4CAF50", edgecolor="white")
            else:
                df.select_dtypes(
                    include=[np.number]
                ).hist(ax=ax, bins=20)

        elif chart_type == "box":
            if x and y and x in df.columns and y in df.columns:
                sns.boxplot(data=df, x=x, y=y,
                            palette="Set2", ax=ax)

        elif chart_type == "heatmap":
            numeric = df.select_dtypes(include=[np.number])
            if not numeric.empty:
                sns.heatmap(
                    numeric.corr(), annot=True,
                    fmt=".2f", cmap="coolwarm", ax=ax
                )

        ax.set_title(title, fontsize=13, fontweight="bold")
        if x:
            ax.set_xlabel(x)
        if y:
            ax.set_ylabel(y)
        plt.tight_layout()

        # Save chart
        safe_title = title.replace(" ", "_")[:30]
        chart_path = os.path.join(
            config.CHARTS_DIR, f"{safe_title}.png"
        )
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()

        return f"✅ Chart saved → {chart_path}"

    except Exception as e:
        plt.close()
        return f"❌ Chart error: {e}"


def run_python(code: str) -> str:
    """
    Executes arbitrary Python code.
    The dataframe is available as 'df'.
    Returns stdout + any errors.
    """
    if _current_df is None:
        return "❌ No data loaded. Call load_csv first."

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    try:
        # Make df available in execution context
        exec_globals = {
            "df": _current_df,
            "pd": pd,
            "np": np,
            "plt": plt,
        }
        exec(code, exec_globals)
        output = buffer.getvalue()
        return output if output else "✅ Code executed (no output)"

    except Exception as e:
        return f"❌ Python error:\n{traceback.format_exc()}"

    finally:
        sys.stdout = old_stdout


def search_web(query: str) -> str:
    """
    Searches the web for context about the query.
    Uses DuckDuckGo's lite HTML interface.
    """
    try:
        url = f"https://lite.duckduckgo.com/lite/?q={requests.utils.quote(query)}"
        r = requests.get(
            url, timeout=8,
            headers={"User-Agent": "ToolMind/1.0 (research)"}
        )
        soup = BeautifulSoup(r.text, "html.parser")

        results = []
        for a in soup.find_all("a", class_="result-link")[:3]:
            results.append(a.get_text(strip=True))

        snippets = []
        for td in soup.find_all("td", class_="result-snippet")[:3]:
            snippets.append(td.get_text(strip=True))

        if not results and not snippets:
            return f"No results found for: {query}"

        output = f"Web search results for: '{query}'\n\n"
        for i, (title, snippet) in enumerate(
            zip(results, snippets), 1
        ):
            output += f"{i}. {title}\n   {snippet}\n\n"

        return output

    except Exception as e:
        return f"❌ Search error: {e}"


def write_report(title: str, content: str) -> str:
    """
    Saves the agent's findings to a markdown file.
    """
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = title.replace(" ", "_")[:40]
    filename = f"{safe_title}_{timestamp}.md"
    path = os.path.join(config.REPORTS_DIR, filename)

    report = (
        f"# {title}\n\n"
        f"*Generated by ToolMind Agent*  \n"
        f"*{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        f"---\n\n"
        f"{content}"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(report)

    return f"✅ Report saved → {path}"


# ── Tool dispatcher ───────────────────────────────────────────

TOOL_MAP = {
    "load_csv":     load_csv,
    "describe_data": describe_data,
    "run_stats":    run_stats,
    "filter_data":  filter_data,
    "plot_chart":   plot_chart,
    "run_python":   run_python,
    "search_web":   search_web,
    "write_report": write_report,
}


def execute_tool(tool_name: str, args: dict) -> str:
    """Dispatches a tool call and returns its result."""
    if tool_name not in TOOL_MAP:
        return f"❌ Unknown tool: {tool_name}"
    try:
        return TOOL_MAP[tool_name](**args)
    except TypeError as e:
        return f"❌ Wrong arguments for {tool_name}: {e}"
    except Exception as e:
        return f"❌ Tool error in {tool_name}: {e}"