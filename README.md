# AI Agent with Tool Use

This project implements an AI agent using LangChain that can utilize external tools to enhance its responses. The agent integrates a **mathematical calculator** and a **web search tool** (DuckDuckGo Search) to perform computations and fetch real-time information. The underlying LLM is **Llama 3.1 70B** hosted on Groq, ensuring high-quality responses.

## Features
- **Web Search:** The agent can retrieve up-to-date information using DuckDuckGo Search.
- **Mathematical Computation:** It can evaluate mathematical expressions using a secure, restricted `eval()` environment.
- **ReAct Agent Framework:** Implements the Reasoning + Acting (ReAct) paradigm for tool use.
- **Error Handling & Limits:** Includes parsing error handling and iteration limits to prevent infinite loops.

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install langchain langchain_core langchain_groq langchain_community tqdm
```

Additionally, set up your **Groq API key** as an environment variable:

```bash
export GROQ_API_KEY='your_api_key_here'
```

## Usage
To run the AI agent, execute the script:

```bash
python script.py
```

You can modify the test queries in the script or call the `run_agent` function with custom input:

```python
response = run_agent("What is the square root of 144 plus 10?")
print(response)
```

## Example Queries

### 1. Web Search + Math Calculation
**Input:**
```
What is the current population of Tokyo and calculate it divided by 1000?
```
**Processing:**
- The agent fetches Tokyo's population from DuckDuckGo Search.
- It performs the division using the Math Calculator tool.

**Output:**
```
The population of Tokyo is approximately 14 million. Dividing it by 1000 gives 14,000.
```

### 2. Pure Math Calculation
**Input:**
```
What is 356 * 289 + the square root of 529?
```
**Output:**
```
Result: 102985.0
```

### 3. General Knowledge
**Input:**
```
Who won the Nobel Prize in Physics in 2023 and what were they known for?
```
**Output:**
```
[Fetches information from DuckDuckGo and returns the answer]
```

## Agent Execution Details
The agent follows these steps:
1. Processes user input.
2. Determines whether to use a tool (Search or Math Calculator).
3. Executes tool actions if necessary.
4. Returns the final response to the user.

## Future Improvements
- Extend the toolset (e.g., Weather API, Wikipedia lookup).
- Improve tool selection logic for multi-step reasoning.
- Optimize response formatting for clarity.

This AI agent demonstrates how an LLM can effectively leverage external tools for more accurate and informed responses.

