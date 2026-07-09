# 🧠 ToolMind — Multi-Tool AI Data Analysis Agent

An autonomous AI agent capable of analyzing datasets through **tool calling**, **memory**, and **self-reflection**. Instead of relying on a single prompt, ToolMind reasons about a task, selects the appropriate tools, evaluates its own progress, and iteratively improves its analysis before producing the final answer.

Built using **Python** and powered by **Groq's LLaMA-3.3-70B**, ToolMind demonstrates how modern AI agents can solve real-world data analysis tasks by combining reasoning with external tools.

---

## ✨ Features

* 🤖 Autonomous tool selection
* 🧠 Memory of previous actions and observations
* 🔄 Self-reflection and iterative reasoning
* 📊 Automated exploratory data analysis
* 📈 Statistical analysis and visualization
* 🌐 Web search for additional context
* 📝 Automatic report generation
* 💻 Python code execution for custom analyses

---

# 🏗️ Architecture

```text
                User Request
                      │
                      ▼
             Natural Language Query
                      │
                      ▼
              LLM Reasoning Engine
          (Groq LLaMA-3.3-70B)
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
     Tool Selection          Memory Lookup
          │                       │
          └───────────┬───────────┘
                      ▼
               Execute Tool
                      │
                      ▼
              Observe Results
                      │
                      ▼
             Self-Reflection Loop
      "Did this move me closer to the answer?"
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
         Yes                     No
          │                       │
          ▼                  Try another tool
     Final Answer              (Max 3 retries)
```

---

# 🛠️ Available Tools

| Tool                 | Description                                            |
| -------------------- | ------------------------------------------------------ |
| 📂 **load_csv**      | Load a CSV dataset into memory                         |
| 📋 **describe_data** | Inspect columns, shape, data types, and missing values |
| 📊 **run_stats**     | Calculate descriptive statistics and correlations      |
| 🔍 **filter_data**   | Filter rows using custom conditions                    |
| 📈 **plot_chart**    | Generate visualizations using Matplotlib               |
| 💻 **run_python**    | Execute arbitrary Python code                          |
| 🌐 **search_web**    | Retrieve contextual information from the web           |
| 📝 **write_report**  | Export findings to a Markdown report                   |

---

# 🧠 Memory System

ToolMind maintains an internal memory throughout each task.

It remembers:

* Previously executed tools
* Tool outputs
* Failed attempts
* Intermediate reasoning
* Current dataset state
* Remaining objectives

This prevents redundant work and enables multi-step reasoning.

---

# 🔄 Self-Reflection

Unlike traditional chatbots, ToolMind evaluates its own progress after every tool execution.

After each step, it asks:

* Did this move me closer to solving the user's request?
* Is the current analysis sufficient?
* Did any tool fail?
* Should I use another tool?
* Have I completely answered the question?

If necessary, the agent retries with a different strategy before producing the final response.

Maximum reflection retries: **3**

---

# 🚀 Example Workflow

### User Prompt

```text
Analyze this sales CSV and identify which region generated the highest revenue.
```

### Agent Reasoning

```text
Step 1
Load the dataset

↓
load_csv("sales.csv")

Step 2
Understand the dataset

↓
describe_data()

Step 3
Compute statistics grouped by region

↓
run_stats(group_by="region")

Step 4
Visualize the results

↓
plot_chart(type="bar", x="region", y="sales")

Step 5
Self-reflection

Did I answer the user's question?

YES ✅

Step 6
Generate report

↓
write_report()
```

---

# 📁 Project Structure

```text
ToolMind/
│
├── agent/
│   ├── memory.py
│   ├── planner.py
│   ├── reflection.py
│   ├── tools.py
│   └── agent.py
│
├── data/
│
├── outputs/
│
├── main.py
├── config.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Tech Stack

* Python
* Groq API
* LLaMA-3.3-70B
* Pandas
* NumPy
* Matplotlib
* Requests

---

# 🎯 Future Improvements

* Multi-agent collaboration
* SQL database tools
* PDF and Excel support
* Retrieval-Augmented Generation (RAG)
* Vector memory
* Code debugging agent
* Automatic dashboard generation
* LangGraph integration
* MCP-compatible tool interface

---

# 📌 Inspiration

ToolMind is inspired by the emerging paradigm of **AI agents** that combine reasoning, memory, planning, and tool use to solve complex real-world problems. Rather than acting as a simple chatbot, the agent autonomously decides which actions to perform, evaluates its own progress, and adapts its strategy until the task is complete.

---

# 📄 License

This project is released under the MIT License.

---

## ⭐ If you found this project interesting, consider giving it a star!
