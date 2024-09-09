import os
import platform
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain_experimental.utilities import PythonREPL
from langchain_community.utilities import StackExchangeAPIWrapper
from langchain_core.tools import Tool

# Function to save API keys to .env
def save_api_key_to_env(key_name, api_key):
    if os.path.exists(".env"):
        with open(".env", "a") as env_file:  # Append mode to keep existing keys
            env_file.write(f"{key_name}={api_key}\n")
    else:
        with open(".env", "w") as env_file:  # Write mode if .env doesn't exist
            env_file.write(f"{key_name}={api_key}\n")
    print(f"{key_name} saved to .env file.")

# Function to get and verify API keys
def get_api_key(key_name):
    # Load environment variables from the .env file
    load_dotenv()

    api_key = os.getenv(key_name)
    if not api_key:
        # Prompt the user to enter the API key if not found
        api_key = input(f"Enter your {key_name}: ")
        save_api_key_to_env(key_name, api_key)
        # Reload environment to use new API key
        load_dotenv()
        api_key = os.getenv(key_name)
    return api_key

# Get API keys
anthropic_api_key = get_api_key("ANTHROPIC_API_KEY")

# Initialize Python REPL tool
python_repl = PythonREPL()

# Create a Tool instance from the PythonREPL tool
python_repl_tool = Tool(
    name="PythonREPL",
    description="A Python shell. Use this to execute Python commands. Input should be a valid Python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run
)

# Initialize StackExchange API Wrapper
stackexchange = StackExchangeAPIWrapper()

# Create a Tool instance for StackExchange API
stackexchange_tool = Tool.from_function(
    func=stackexchange.run,
    name="StackExchangeAPI",
    description="A tool to get related questions from StackExchange for code-related errors or general queries."
)

# Create the agent
memory = MemorySaver()
model = ChatAnthropic(model_name="claude-3-sonnet-20240229", api_key=anthropic_api_key)

# Combine tools (Python REPL and StackExchange)
tools = [python_repl_tool, stackexchange_tool]

# Create the agent executor with tools and memory
agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Configuration for the agent
config = {"configurable": {"thread_id": "abc123"}}

# Use the agent in a loop for accepting Python scripts or StackExchange queries
while True:
    prompt = input("Enter a prompt (type 'exit' to quit): ")

    if prompt.lower() == "exit":
        print("Exiting the agent loop.")
        break

    try:
        # Pass the prompt to the agent for processing
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=prompt)]}, config
        ):
            if hasattr(chunk, 'tools'):
                # Handle tool usage
                tool_message = chunk.tools.messages[0]
                tool_name = tool_message.name
                code_snippet = tool_message.content

                if tool_name == "PythonREPL":
                    # Execute the code in the Python REPL
                    result = python_repl.run(code_snippet)
                    print(result)
                elif tool_name == "StackExchangeAPI":
                    # Retrieve the StackExchange question
                    question = stackexchange.run(code_snippet)
                    print(question)
            else:
                # Handle standard agent output
                print(chunk)
        print("----")
    except Exception as e:
        print(f"Error: {e}")
