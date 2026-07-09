"""
ToolMind — Multi-Tool Data Analysis Agent

An agent with tool use, memory, and self-reflection
that autonomously analyzes data and produces reports.

Usage:
  python main.py
  python main.py --task "Your custom analysis task"
"""

import os
import argparse
from dotenv import load_dotenv
load_dotenv()

import config
from agent.executor import run_agent

os.makedirs(config.CHARTS_DIR, exist_ok=True)
os.makedirs(config.REPORTS_DIR, exist_ok=True)
os.makedirs(config.DATA_DIR, exist_ok=True)

# Sample tasks to demo the agent
SAMPLE_TASKS = [
    "Load sample_sales.csv and tell me which region "
    "had the highest total revenue. Show a bar chart "
    "and write a report with your findings.",

    "Load sample_sales.csv, find the top performing "
    "salesperson by total revenue, analyze their "
    "product mix, and write a detailed report.",

    "Load sample_sales.csv and analyze the relationship "
    "between customer rating and return rate. "
    "Show a scatter plot and explain what you find.",

    "Load sample_sales.csv and find which product "
    "has the highest revenue per unit sold. "
    "Compare across regions and write a report.",
]


def create_sample_data():
    import pandas as pd
    import numpy as np

    path = os.path.join(config.DATA_DIR, "sample_sales.csv")
    if os.path.exists(path):
        return

    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "date": pd.date_range(
            "2024-01-01", periods=n, freq="D"
        ).strftime("%Y-%m-%d"),
        "region": np.random.choice(
            ["North", "South", "East", "West"], n
        ),
        "product": np.random.choice(
            ["Laptop", "Phone", "Tablet", "Watch"], n
        ),
        "salesperson": np.random.choice(
            ["Alice", "Bob", "Carol", "David", "Eve"], n
        ),
        "units_sold": np.random.randint(1, 50, n),
        "unit_price": np.random.choice(
            [299, 499, 799, 1299], n
        ),
        "customer_rating": np.round(
            np.random.uniform(3.0, 5.0, n), 1
        ),
        "return_rate": np.round(
            np.random.uniform(0.0, 0.15, n), 3
        ),
    })
    df["revenue"] = df["units_sold"] * df["unit_price"]
    df.to_csv(path, index=False)
    print(f"✅ Sample data → {path}")

while True:
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--task", type=str,
            default=SAMPLE_TASKS[0],
            help="Task for the agent to solve"
        )
        parser.add_argument(
            "--all-tasks", action="store_true",
            help="Run all sample tasks"
        )
        args = parser.parse_args()

        create_sample_data()

        if args.all_tasks:
            for i, task in enumerate(SAMPLE_TASKS, 1):
                print(f"\n{'#'*60}")
                print(f"  Task {i}/{len(SAMPLE_TASKS)}")
                print(f"{'#'*60}")
                answer = run_agent(task)
                print(f"\nFinal Answer: {answer}")
        else:
            answer = run_agent(args.task)
            print(f"\nFinal Answer: {answer}")