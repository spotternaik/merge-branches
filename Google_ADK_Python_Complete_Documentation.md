# Google Agent Development Kit (ADK) - Python Complete Documentation

**Version:** 0.2.0  
**Language:** Python 3.9+  
**Last Updated:** October 2025  
**Official Repository:** https://github.com/google/adk-python  
**Documentation:** https://google.github.io/adk-docs/

---

## Table of Contents

1. [Introduction](#introduction)
2. [What is ADK?](#what-is-adk)
3. [Python Quick Installation](#python-quick-installation)
4. [Python Getting Started](#python-getting-started)
5. [Core Concepts](#core-concepts)
6. [Agent Types in Python](#agent-types-in-python)
7. [LLM Agents Deep Dive](#llm-agents-deep-dive)
8. [Python Function Tools](#python-function-tools)
9. [Building Your First Agent](#building-your-first-agent)
10. [Multi-Tool Agents](#multi-tool-agents)
11. [Streaming Agents with Python](#streaming-agents-with-python)
12. [Sessions & Memory Management](#sessions--memory-management)
13. [Running & Testing](#running--testing)
14. [Deployment](#deployment)
15. [Python API Reference](#python-api-reference)

---

## Introduction

**Agent Development Kit (ADK)** is a flexible and modular framework for developing and deploying AI agents. The Python implementation of ADK provides an intuitive, Pythonic API for building conversational and non-conversational agents powered by Large Language Models.

### Why Choose ADK for Python?

- **Developer-friendly:** Write agents like you write regular Python applications
- **Model-agnostic:** Works with Gemini, OpenAI, and other LLMs
- **Rapid development:** Go from concept to working agent in minutes
- **Production-ready:** Designed for local development and cloud deployment
- **Extensible:** Create custom tools, agents, and integration patterns

---

## What is ADK?

ADK is a comprehensive framework that empowers Python developers to:

- **Build** AI-powered agents for conversational and non-conversational tasks
- **Manage** complex agent workflows and multi-agent systems
- **Evaluate** agent performance against test cases and criteria
- **Deploy** agents to various cloud and on-premises environments

### Key Python Features

- **Native Python functions as tools** - No need for complex wrapper classes
- **Async/await support** - Build responsive, high-performance agents
- **Pydantic integration** - Type-safe schemas for inputs and outputs
- **Familiar patterns** - Works like standard Python libraries

---

## Python Quick Installation

### Prerequisites

- **Python:** 3.9 or later
- **pip:** Python package manager
- **Virtual environment (recommended):** `venv` or `conda`

### Step 1: Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# macOS/Linux
source .venv/bin/activate

# Windows CMD
.venv\Scripts\activate.bat

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### Step 2: Install ADK

```bash
pip install google-adk
```

### Step 3: Verify Installation

```bash
python -c "import google.adk; print('ADK installed successfully!')"
```

---

## Python Getting Started

### Project Setup with `adk create`

The fastest way to start is using the ADK CLI:

```bash
adk create my_agent
```

This generates the following structure:

```
my_agent/
    agent.py           # Main agent code
    .env               # API keys and configuration
    __init__.py        # Python package marker
```

### Minimal Python Agent

Here's the simplest working agent:

```python
from google.adk.agents import Agent

root_agent = Agent(
    model='gemini-2.0-flash',
    name='hello_agent',
    description="A simple greeting agent",
    instruction="You are a helpful assistant that greets users warmly."
)
```

### Running Your Agent

**Option 1: Web UI (Recommended for development)**

```bash
adk web my_agent
```

Open `http://localhost:8000` in your browser.

**Option 2: Command-line Interface**

```bash
adk run my_agent
```

**Option 3: API Server**

```bash
adk api_server my_agent
```

---

## Core Concepts

### Agent

The fundamental execution unit in ADK. An agent receives input, uses tools and reasoning, and produces output. In Python, agents are instances of `Agent`, `LlmAgent`, or custom `BaseAgent` subclasses.

### Tools

Python functions or methods that agents can call to perform actions beyond reasoning. Tools are automatically wrapped and exposed to the LLM.

### Session

Represents a single conversation context. Sessions maintain:
- **Events:** Conversation history (user messages, agent responses, tool calls)
- **State:** Working memory for the session
- **Metadata:** Session configuration and metadata

### State Management

- **Session State:** Short-term memory within a single conversation
- **Memory Service:** Long-term storage across multiple sessions (optional)
- **Artifacts:** Manage files and binary data

### Runner

The execution engine that orchestrates agent interactions. In Python:
- `InMemoryRunner` - For local development and testing
- `Runner` - For production deployments

### Events

The basic unit of communication representing what happens during a session:
- User messages
- Agent responses
- Tool calls and results
- Internal state changes

---

## Agent Types in Python

### 1. LLM Agents

Large Language Model-powered agents with dynamic decision-making:

```python
from google.adk.agents import Agent, LlmAgent
from google.genai import types

# Using the Agent alias
agent = Agent(
    model="gemini-2.0-flash",
    name="my_agent",
    description="Description of what the agent does",
    instruction="Detailed instructions for the agent's behavior",
    tools=[...]  # Optional tools
)

# Using LlmAgent explicitly
agent = LlmAgent(
    model="gemini-2.0-flash",
    name="my_agent",
    description="Description of what the agent does",
    instruction="Detailed instructions for the agent's behavior",
    tools=[...]  # Optional tools
)
```

### 2. Workflow Agents

Deterministic agents for structured workflows:

```python
from google.adk.agents import SequentialAgent, ParallelAgent, LoopAgent

# Sequential: Execute agents one after another
seq_agent = SequentialAgent(
    agents=[agent1, agent2, agent3]
)

# Parallel: Execute agents simultaneously
par_agent = ParallelAgent(
    agents=[agent1, agent2, agent3]
)

# Loop: Execute agent(s) until condition is met
loop_agent = LoopAgent(
    agent=processing_agent,
    condition_agent=checker_agent
)
```

### 3. Custom Agents

Extend `BaseAgent` for specialized implementations:

```python
from google.adk.agents import BaseAgent

class CustomAgent(BaseAgent):
    async def run(self, content, context):
        # Your custom logic here
        return result
```

---

## LLM Agents Deep Dive

### Defining Agent Identity

Every LLM agent requires:

```python
from google.adk.agents import Agent

agent = Agent(
    # Required: Model to use
    model="gemini-2.0-flash",
    
    # Required: Unique identifier
    name="capital_agent",
    
    # Recommended: Description for multi-agent systems
    description="Provides capital cities of countries",
    
    # Required: Behavior guidelines
    instruction="""You are a geography expert. When asked for a capital:
1. Identify the country
2. Use the get_capital_city tool if available
3. Return the capital name
""",
    
    # Optional: Tools the agent can use
    tools=[get_capital_city]
)
```

### Instructions with Dynamic State

Instructions can reference session state using template syntax:

```python
instruction = """You are a helpful assistant for user {username}.
You have access to their previous purchases: {purchases}
Always maintain a {tone} tone."""

agent = Agent(
    model="gemini-2.0-flash",
    name="customer_agent",
    instruction=instruction,
    tools=[...]
)
```

### Configuring LLM Generation

Control LLM behavior with `generate_content_config`:

```python
from google.genai import types

agent = Agent(
    model="gemini-2.0-flash",
    name="my_agent",
    instruction="Your instruction here",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2,           # More deterministic
        max_output_tokens=500,     # Response length
        top_p=0.95,               # Nucleus sampling
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            )
        ]
    )
)
```

### Structured Input & Output

Use Pydantic models for type safety:

```python
from pydantic import BaseModel, Field
from google.adk.agents import Agent

class CountryInput(BaseModel):
    country: str = Field(description="Country name")

class CapitalOutput(BaseModel):
    capital: str = Field(description="Capital city")
    population: int = Field(description="Capital population")

agent = Agent(
    model="gemini-2.0-flash",
    name="capital_agent",
    instruction="Respond with JSON matching the output schema",
    input_schema=CountryInput,
    output_schema=CapitalOutput,
    output_key="capital_result"  # Store in session.state["capital_result"]
)
```

### Advanced: Planning with Built-in Planner

Enable multi-step reasoning using Gemini's thinking capability:

```python
from google.adk.planners import BuiltInPlanner
from google.genai import types
from google.adk.agents import Agent

thinking_config = types.ThinkingConfig(
    include_thoughts=True,
    thinking_budget=256
)

agent = Agent(
    model="gemini-2.5-pro-preview",
    name="reasoning_agent",
    instruction="Think carefully about the problem",
    planner=BuiltInPlanner(thinking_config=thinking_config),
    tools=[...]
)
```

### Using ReAct Planning

For models without native thinking:

```python
from google.adk.planners import PlanReActPlanner
from google.adk.agents import Agent

agent = Agent(
    model="gemini-2.0-flash",
    name="reasoning_agent",
    instruction="Think step-by-step",
    planner=PlanReActPlanner(),
    tools=[...]
)
```

---

## Python Function Tools

### Automatic Tool Wrapping

Python functions are automatically wrapped as tools:

```python
def get_weather(city: str) -> dict:
    """Retrieves current weather for a city.
    
    Args:
        city: The name of the city
        
    Returns:
        dict: Weather information
    """
    # Implementation
    return {
        "status": "success",
        "city": city,
        "temperature": 25,
        "condition": "sunny"
    }

agent = Agent(
    model="gemini-2.0-flash",
    name="weather_agent",
    instruction="Provide weather information using the tool",
    tools=[get_weather]  # Function automatically wrapped!
)
```

### Function Signature Best Practices

```python
# ✅ Good: Required and optional parameters
def search_flights(
    destination: str,
    departure_date: str,
    flexible_days: int = 0
) -> dict:
    """Search for flights.
    
    Args:
        destination: Target city
        departure_date: YYYY-MM-DD format
        flexible_days: Days to search before/after (optional)
    """
    pass

# ✅ Good: Optional with typing.Optional
from typing import Optional

def create_profile(
    username: str,
    bio: Optional[str] = None
) -> dict:
    """Create user profile."""
    pass

# ❌ Avoid: *args and **kwargs (not sent to LLM)
def bad_tool(*args, **kwargs):
    """LLM won't understand this."""
    pass
```

### Return Value Guidelines

Tools must return dictionaries with clear structure:

```python
def successful_operation() -> dict:
    return {
        "status": "success",
        "result": "operation completed",
        "data": {...}
    }

def failed_operation() -> dict:
    return {
        "status": "error",
        "error_message": "descriptive error for LLM",
        "details": {...}
    }

# Non-dict return values are wrapped:
def simple_tool() -> str:
    return "result"  # Wrapped as {"result": "result"}
```

### Passing Data Between Tools

Use session state with `temp:` prefix:

```python
def first_tool(query: str) -> dict:
    """First tool stores result in temp state"""
    from google.adk.runtime.invocation_context import get_invocation_context
    
    context = get_invocation_context()
    result = process_query(query)
    context.state["temp:search_result"] = result
    return {"status": "success", "query_processed": True}

def second_tool() -> dict:
    """Second tool reads from temp state"""
    from google.adk.runtime.invocation_context import get_invocation_context
    
    context = get_invocation_context()
    previous_result = context.state.get("temp:search_result")
    # Use previous_result
    return {"status": "success", "used_previous": True}
```

### Long-Running Tools

For operations that take time:

```python
from google.adk.tools import LongRunningFunctionTool

def submit_approval_request(purpose: str, amount: float) -> dict:
    """Submit for approval (returns immediately with ticket ID)"""
    ticket_id = create_ticket(purpose, amount)
    send_notification_to_approver(ticket_id)
    return {
        "status": "pending",
        "ticket_id": ticket_id,
        "purpose": purpose,
        "amount": amount
    }

long_running_tool = LongRunningFunctionTool(func=submit_approval_request)

agent = Agent(
    model="gemini-2.0-flash",
    name="approval_agent",
    instruction="Request approvals for expenses",
    tools=[long_running_tool]
)
```

---

## Building Your First Agent

### Complete Simple Agent Example

```python
# agent.py
from google.adk.agents import Agent

def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.
    
    Args:
        city (str): The name of the city
        
    Returns:
        dict: status and time information
    """
    import datetime
    from zoneinfo import ZoneInfo
    
    timezones = {
        "new york": "America/New_York",
        "london": "Europe/London",
        "tokyo": "Asia/Tokyo",
    }
    
    tz_name = timezones.get(city.lower())
    if not tz_name:
        return {
            "status": "error",
            "error_message": f"Timezone for {city} not available"
        }
    
    tz = ZoneInfo(tz_name)
    now = datetime.datetime.now(tz)
    return {
        "status": "success",
        "city": city,
        "time": now.strftime("%Y-%m-%d %H:%M:%S %Z")
    }

root_agent = Agent(
    model='gemini-2.0-flash',
    name='time_agent',
    description="Tells the current time in specified cities",
    instruction="""You are a helpful assistant that provides current time.
When asked for the time in a city:
1. Use the get_current_time tool
2. Report the exact time found
3. Be friendly and conversational""",
    tools=[get_current_time]
)
```

### Running It

```bash
adk web my_agent
# or
adk run my_agent
```

---

## Multi-Tool Agents

### Project Structure

```
multi_tool_agent/
    __init__.py
    agent.py
    .env
```

### Complete Multi-Tool Example

```python
# agent.py
import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent

def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.
    
    Args:
        city (str): The name of the city
        
    Returns:
        dict: Weather status and report
    """
    if city.lower() == "new york":
        return {
            "status": "success",
            "report": (
                "The weather in New York is sunny with a temperature of 25 degrees"
                " Celsius (77 degrees Fahrenheit)."
            ),
        }
    else:
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available.",
        }

def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.
    
    Args:
        city (str): The name of the city
        
    Returns:
        dict: Time status and report
    """
    if city.lower() == "new york":
        tz_identifier = "America/New_York"
    else:
        return {
            "status": "error",
            "error_message": f"Sorry, I don't have timezone information for {city}.",
        }
    
    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    report = (
        f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
    )
    return {"status": "success", "report": report}

root_agent = Agent(
    name="weather_time_agent",
    model="gemini-2.0-flash",
    description=(
        "Agent to answer questions about the time and weather in a city."
    ),
    instruction=(
        "You are a helpful agent who can answer user questions about the time "
        "and weather in a city. Use the available tools to provide accurate information."
    ),
    tools=[get_weather, get_current_time],
)
```

### Configuration (`.env`)

```
# Using Google AI Studio
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_API_KEY=PASTE_YOUR_API_KEY

# OR using Google Cloud Vertex AI
GOOGLE_GENAI_USE_VERTEXAI=TRUE
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
```

### Example Prompts to Try

```
- What is the weather in New York?
- What is the time in New York?
- Tell me about the weather and current time in New York
- How's the weather today?
```

---

## Streaming Agents with Python

### Supported Models for Streaming

Check for models supporting Gemini Live API:
- Google AI Studio: https://ai.google.dev/gemini-api/docs/models#live-api
- Vertex AI: https://cloud.google.com/vertex-ai/generative-ai/docs/live-api

### Basic Streaming Agent

```python
# agent.py
from google.adk.agents import Agent
from google.adk.tools import google_search

root_agent = Agent(
    name="basic_search_agent",
    model="gemini-2.0-flash-live-001",  # Live model required
    description="Agent to answer questions using Google Search",
    instruction="You are an expert researcher. Always stick to the facts.",
    tools=[google_search]  # Built-in grounding tool
)
```

### Project Structure for Streaming

```
adk-streaming/
└── app/
    ├── .env
    └── google_search_agent/
        ├── __init__.py
        └── agent.py
```

### Setting Up SSL for Voice/Video

```bash
cd app

# macOS/Linux
export SSL_CERT_FILE=$(python -m certifi)

# Windows PowerShell
$env:SSL_CERT_FILE = (python -m certifi)
```

### Running Streaming Agent

```bash
cd app
adk web
```

Features available:
- Text input/output
- Real-time audio input/output
- Video input with description
- Microphone and camera buttons in web UI

---

## Sessions & Memory Management

### InMemorySessionService (Development)

```python
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Create session service
session_service = InMemorySessionService()

# Create session
session = await session_service.create_session(
    app_name="my_app",
    user_id="user123",
    session_id="session456"
)

# Access session data
session.state["key"] = "value"
session.id  # Session identifier
```

### Runner with Session Management

```python
from google.adk.runners import Runner
from google.genai import types

runner = Runner(
    agent=my_agent,
    app_name="my_app",
    session_service=session_service
)

# Run agent
user_message = types.Content(
    role='user',
    parts=[types.Part(text="What's the weather?")]
)

async for event in runner.run_async(
    user_id="user123",
    session_id="session456",
    new_message=user_message
):
    if event.is_final_response():
        print(f"Agent: {event.content.parts[0].text}")
```

### Accessing Session State

```python
# Get current session
session = await session_service.get_session(
    app_name="my_app",
    user_id="user123",
    session_id="session456"
)

# Read state
user_data = session.state.get("user_preferences")

# Write state
session.state["conversation_count"] = 5
```

---

## Running & Testing

### Running Agents

**Web UI (Interactive Development)**
```bash
adk web my_agent
```
- Access at http://localhost:8000
- Inspect events and traces
- Test with text and voice

**Command Line Interface**
```bash
adk run my_agent
```
- Direct terminal interaction
- Pipe input: `echo "query" | adk run my_agent`

**API Server**
```bash
adk api_server my_agent
```
- FastAPI server for testing
- HTTP endpoints for agent invocation

### Programmatic Testing

```python
import asyncio
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

async def test_agent():
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="test_app",
        user_id="test_user"
    )
    
    runner = Runner(
        agent=my_agent,
        app_name="test_app",
        session_service=session_service
    )
    
    message = types.Content(
        role='user',
        parts=[types.Part(text="Hello, agent!")]
    )
    
    final_response = None
    async for event in runner.run_async(
        user_id="test_user",
        session_id=session.id,
        new_message=message
    ):
        if event.is_final_response():
            final_response = event.content.parts[0].text
    
    assert final_response is not None
    print(f"Test passed: {final_response}")

# Run test
asyncio.run(test_agent())
```

### Debugging with Events

```python
async for event in runner.run_async(...):
    # Event information
    print(f"Author: {event.author}")
    print(f"Event type: {event.type}")
    print(f"Is function call: {event.is_function_call()}")
    print(f"Is final response: {event.is_final_response()}")
    
    # Extract content
    if event.content:
        for part in event.content.parts:
            if part.text:
                print(f"Text: {part.text}")
            if part.function_call:
                print(f"Tool call: {part.function_call.name}")
```

---

## Deployment

### Docker Containerization

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["adk", "api_server", "my_agent"]
```

### Cloud Run Deployment

```bash
# Build and deploy to Cloud Run
gcloud run deploy my-adk-agent \
    --source . \
    --platform managed \
    --region us-central1 \
    --memory 512Mi \
    --allow-unauthenticated
```

### Vertex AI Agent Engine

Deploy managed agents with built-in scaling:

```python
# Coming soon: Full Vertex AI Agent Engine integration
# ADK handles session management automatically
```

---

## Python API Reference

### Core Classes

**Agent / LlmAgent**

```python
from google.adk.agents import Agent, LlmAgent

agent = Agent(
    model: str,                          # Required: LLM model ID
    name: str,                           # Required: Agent identifier
    description: str = None,             # Optional: For multi-agent routing
    instruction: str = None,             # Optional: Behavior guidelines
    tools: List[Tool] = None,            # Optional: Available tools
    generate_content_config = None,      # Optional: LLM parameters
    input_schema = None,                 # Optional: Input structure
    output_schema = None,                # Optional: Output structure
    output_key: str = None,              # Optional: State key for output
    include_contents = 'default',        # Optional: Include history
    planner = None,                      # Optional: BuiltInPlanner, PlanReActPlanner
    code_executor = None,                # Optional: Code execution
)
```

**Runner**

```python
from google.adk.runners import Runner

runner = Runner(
    agent: BaseAgent,                    # Required: Agent to run
    app_name: str,                       # Required: Application name
    session_service: SessionService,     # Required: Session management
)

# Methods
async for event in runner.run_async(
    user_id: str,
    session_id: str,
    new_message: Content
) -> Event

events = runner.run(
    user_id: str,
    session_id: str,
    new_message: Content
) -> Iterable[Event]
```

**InMemorySessionService**

```python
from google.adk.sessions import InMemorySessionService

service = InMemorySessionService()

# Methods
session = await service.create_session(
    app_name: str,
    user_id: str,
    session_id: str = None
) -> Session

session = await service.get_session(
    app_name: str,
    user_id: str,
    session_id: str
) -> Session
```

**Tools**

```python
# Function tools (automatic wrapping)
def my_tool(param: str) -> dict:
    """Docstring becomes tool description"""
    return {"result": "value"}

# Long-running tools
from google.adk.tools import LongRunningFunctionTool

long_tool = LongRunningFunctionTool(func=my_function)

# Built-in tools
from google.adk.tools import google_search
```

**Planners**

```python
from google.adk.planners import BuiltInPlanner, PlanReActPlanner
from google.genai import types

# Built-in thinking
planner = BuiltInPlanner(
    thinking_config=types.ThinkingConfig(
        include_thoughts=True,
        thinking_budget=256
    )
)

# ReAct planning
planner = PlanReActPlanner()
```

### Imports

```python
# Agents
from google.adk.agents import Agent, LlmAgent, SequentialAgent, ParallelAgent, LoopAgent, BaseAgent

# Runners
from google.adk.runners import Runner, InMemoryRunner

# Sessions
from google.adk.sessions import Session, InMemorySessionService

# Tools
from google.adk.tools import google_search, LongRunningFunctionTool, FunctionTool

# Planners
from google.adk.planners import BuiltInPlanner, PlanReActPlanner

# Types (from genai)
from google.genai import types
from google.genai.types import Content, Part, GenerateContentConfig

# State/Context
from google.adk.runtime.invocation_context import get_invocation_context

# Artifacts
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
```

---

## Complete End-to-End Example

```python
# Complete example with async patterns
import asyncio
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Define tools
def calculate_sum(numbers: list) -> dict:
    """Calculate the sum of a list of numbers."""
    total = sum(numbers)
    return {
        "status": "success",
        "total": total,
        "count": len(numbers),
        "average": total / len(numbers) if numbers else 0
    }

def format_currency(amount: float, currency: str = "USD") -> dict:
    """Format amount as currency."""
    return {
        "status": "success",
        "formatted": f"{currency} {amount:,.2f}"
    }

# Create agent
calculator_agent = Agent(
    model="gemini-2.0-flash",
    name="calculator",
    description="Performs calculations and formatting",
    instruction="""You are a helpful calculator. When given numbers:
1. Sum them if asked for total
2. Format currency results appropriately
3. Provide clear explanations""",
    tools=[calculate_sum, format_currency]
)

# Main execution
async def main():
    # Setup
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="calc_app",
        user_id="user1"
    )
    
    runner = Runner(
        agent=calculator_agent,
        app_name="calc_app",
        session_service=session_service
    )
    
    # Send queries
    queries = [
        "What's the sum of 10, 20, 30?",
        "Format 1500.50 as USD currency"
    ]
    
    for query in queries:
        print(f"\nUser: {query}")
        
        message = types.Content(
            role='user',
            parts=[types.Part(text=query)]
        )
        
        async for event in runner.run_async(
            user_id="user1",
            session_id=session.id,
            new_message=message
        ):
            if event.is_final_response():
                response = event.content.parts[0].text
                print(f"Agent: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Best Practices for Python ADK Development

### 1. Always Use Virtual Environments

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

### 2. Type Hints and Docstrings

```python
def tool_function(
    input_param: str,
    optional_param: int = 10
) -> dict:
    """Clear, detailed description of what the tool does.
    
    Args:
        input_param: Description of the input
        optional_param: Optional parameter description
        
    Returns:
        dict: Description of return value structure
    """
    pass
```

### 3. Error Handling in Tools

```python
def robust_tool(query: str) -> dict:
    """Tool with proper error handling."""
    try:
        result = perform_operation(query)
        return {
            "status": "success",
            "result": result
        }
    except ValueError as e:
        return {
            "status": "error",
            "error_message": str(e)
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Unexpected error: {str(e)}"
        }
```

### 4. Environment Configuration

Store secrets in `.env` file:

```
GOOGLE_API_KEY=your_key_here
GOOGLE_CLOUD_PROJECT=project-id
DATABASE_URL=connection_string
```

Load with `python-dotenv`:

```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
```

### 5. Async Patterns

Always use async/await for I/O operations:

```python
import aiohttp
import asyncio

async def fetch_data(url: str) -> dict:
    """Asynchronous HTTP request."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.json()
            return {"status": "success", "data": data}
```

---

## Resources & Links

### Official Documentation
- **Main ADK Docs:** https://google.github.io/adk-docs/
- **Python SDK:** https://github.com/google/adk-python
- **Code Samples:** https://github.com/google/adk-samples

### Related Documentation
- **Gemini API:** https://ai.google.dev/
- **Vertex AI:** https://cloud.google.com/vertex-ai
- **Pydantic:** https://docs.pydantic.dev/

### Community
- **GitHub Issues:** Report bugs and request features
- **Stack Overflow:** Tag: `google-adk` or `agent-development-kit`
- **Google Cloud Community:** General discussions

---

## Troubleshooting

### Common Issues

**Issue: "ModuleNotFoundError: No module named 'google.adk'"**
```bash
# Solution: Install ADK
pip install google-adk
```

**Issue: API authentication fails**
```bash
# Solution: Set API key or authenticate with gcloud
export GOOGLE_API_KEY=your_key
# or
gcloud auth application-default login
```

**Issue: Agent not appearing in `adk web` dropdown**
```bash
# Solution: Run adk web from parent directory of agent folder
cd parent_directory
adk web
```

**Issue: "_make_subprocess_transport NotImplementedError" on Windows**
```bash
# Solution: Use --no-reload flag
adk web --no-reload
```

---

## Glossary (Python-specific)

- **Agent:** Instance of Agent or LlmAgent class
- **Tool:** Python function wrapped as FunctionTool
- **Runner:** Executor managing agent lifecycle
- **Session:** Conversation context with state
- **Event:** Represents occurrence in agent execution
- **LLM:** Language model (e.g., "gemini-2.0-flash")
- **Invocation Context:** Execution context for tool calls
- **Streaming:** Bidirectional real-time communication

---

**Last Updated:** October 2025  
**Python Version Support:** 3.9 - 3.13  
**License:** Apache License 2.0

Copyright © Google 2025

