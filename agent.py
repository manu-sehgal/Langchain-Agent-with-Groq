import os
from typing import List, Union, Any, Optional
from langchain_core.tools import BaseTool, Tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain.agents.react.output_parser import ReActOutputParser
from langchain_core.runnables import RunnablePassthrough
import math


# Set the Groq API key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")  # Use environment variable

# Initialize the Groq LLM with Llama 3.1 70B
llm = ChatGroq(
    model="llama3-70b-8192",  # Llama 3.1 70B model on Groq
    temperature=0.1,  # Lower temperature for more deterministic outputs
    max_tokens=1024,  # Adjust as needed
)

# Create DuckDuckGo Search Tool
search_tool = DuckDuckGoSearchRun()


# Create Math Calculator Tool with type annotations
class MathCalculator(BaseTool):
    name: str = "Math Calculator"
    description: str = (
        "Useful for performing mathematical calculations. Input should be a valid mathematical expression."
    )

    def _run(self, query: str) -> str:
        try:
            # Safely evaluate the mathematical expression
            # Using a restricted set of functions from the math module
            allowed_names = {
                k: v
                for k, v in math.__dict__.items()
                if not k.startswith("__")
                and (callable(v) or isinstance(v, (int, float)))
            }
            allowed_names.update(
                {
                    "abs": abs,
                    "round": round,
                    "max": max,
                    "min": min,
                    "sum": sum,
                    "pow": pow,
                    "int": int,
                    "float": float,
                }
            )

            # Safely evaluate the expression
            result = eval(query, {"__builtins__": {}}, allowed_names)
            return f"Result: {result}"
        except Exception as e:
            return f"Error in calculation: {str(e)}"

    async def _arun(self, query: str) -> str:
        return self._run(query)


# Initialize the calculator tool
calculator_tool = MathCalculator()

# List of tools for the agent to use
tools = [search_tool, calculator_tool]

# Create a prompt for the agent
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant that can use tools to answer questions.
    You have access to the following tools:
    
    {tools}
    
    To use a tool, use the following format:
    
    Action: the action to take, must be one of {tool_names}
    Action Input: the input to the action
    Observation: the result of the action
    
    When you have gathered enough information to answer the question, respond directly.
    Always think step-by-step about what information you need to answer the user's question.
    If you need to search for information, use the DuckDuckGo Search tool.
    If you need to perform mathematical calculations, use the Math Calculator tool.

    Make sure to only give to-the-point answers.
    """,
        ),
        ("human", "{input}"),
        ("ai", "{agent_scratchpad}"),
    ]
)

# Create the ReAct agent
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,  # Limit the number of iterations to prevent infinite loops
)


# Example function to run the agent
def run_agent(user_input: str) -> str:
    """Run the agent with the given user input."""
    try:
        result = agent_executor.invoke({"input": user_input})
        return result["output"]
    except Exception as e:
        return f"Error executing agent: {str(e)}"


# Example usage
if __name__ == "__main__":
    # Example queries to test the agent
    test_queries = [
        "What is the current population of Tokyo and calculate it divided by 1000?",
        # "What is 356 * 289 + the square root of 529?",
        # "Who won the Nobel Prize in Physics in 2023 and what were they known for?"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        response = run_agent(query)
        print(f"Response: {response}")
        print("=" * 50)
