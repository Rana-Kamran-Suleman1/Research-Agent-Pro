from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import  HumanMessage,AIMessage,ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langchain.agents.middleware import (
    wrap_tool_call,
    ToolRetryMiddleware,
    ModelRetryMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    HumanInTheLoopMiddleware
)

from langchain_community.tools import (
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
    ArxivQueryRun
)
from langchain_community.utilities import(
    DuckDuckGoSearchAPIWrapper,
    WikipediaAPIWrapper,
    ArxivAPIWrapper
)


#############################################        TOOLS
search_wrapper=DuckDuckGoSearchAPIWrapper(max_results=5)
search_tool=DuckDuckGoSearchRun(
    api_wrapper=search_wrapper,
    name="web_search",
   description="Search the web using DuckDuckGo for current information, news, and general web content. Use this when you need up-to-date information or content not available on Wikipedia."
)

wiki_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)
wiki_tool=WikipediaQueryRun(
    api_wrapper=wiki_wrapper,
    name="wikipedia",
   description="Search Wikipedia for encyclopedia-style information, facts, and summaries. Use this for quick factual queries and well-established knowledge."
)
arxiv_wrapper=ArxivAPIWrapper(top_k_results=2,doc_content_chars_max=2000)
arxiv_tool=ArxivQueryRun(
    api_wrapper=arxiv_wrapper,
    name="arxiv",
   description="Search arXiv for academic papers, scientific research, and scholarly articles. Use this for technical and academic research queries."
)

tools=[search_tool,wiki_tool,arxiv_tool]

###############################################  MIDDLEWARE
@wrap_tool_call
def tool_handle_error(request, handler):
    try:
        return handler(request)
    except Exception as err:
        print(f"Error:  {err}")

tool_retry=ToolRetryMiddleware(
    max_retries=2,
    max_delay=60,
    backoff_factor=1.5, ### Multiple with previous delay
    tools=["web_search","wikipedia","arxiv"],
    on_failure="continue"
)
model_retry=ModelRetryMiddleware(
    max_retries=2,
    max_delay=60,
    backoff_factor=1.5,
    on_failure="continue"
)

human_in_loop=HumanInTheLoopMiddleware(
    interrupt_on={"web_search": {
                    "allowed_decisions": ["approve", "reject"] }}
    )

# )
tool_call_limit=ToolCallLimitMiddleware(
    run_limit=5
)

middleware=[tool_handle_error,tool_retry,model_retry,tool_call_limit,human_in_loop]


#################################################  CREATE AGENT

SYSTEM_RESEARCH_PROMPT = '''You are a Research AI Agent, an intelligent assistant specialized in conducting research and gathering information from multiple sources.

Your capabilities:
- Web Search: Use DuckDuckGo to find current information, news, and web content
- Wikipedia: Query for encyclopedia-style facts and well-established knowledge
- arXiv: Search for academic papers and scientific research
- DateTime: Get the current date and time when needed

Guidelines:
1. Always use the most appropriate tool for the type of information needed
2. For factual queries, start with Wikipedia
3. For current events or recent information, use web search
4. For academic or technical research, use arXiv
5. Synthesize information from multiple sources when possible
6. Provide clear, well-structured responses with proper citations
7. If a query is ambiguous, ask for clarification before searching

When responding, cite your sources and provide accurate, up-to-date information.
'''
memory=MemorySaver()

def create_research_agent():
    llm=ChatOllama(
    model="qwen3.5:cloud",
    temperature=0
    )

    agent=create_agent(
        model=llm,
        name="",
        tools=tools,
        middleware=middleware,
        system_prompt=SYSTEM_RESEARCH_PROMPT,
        checkpointer=memory
    )

    return agent


def banner():
    """Display the agent banner."""
    print("Research AI Agent")

def stream_response(agent, query: str, config: dict):
    """Stream and print the agent's response to a query.

    Args:
        agent: The LangChain agent instance.
        query: The user's query string.
        config: Configuration dictionary with thread_id for memory.
    """
    for chunk in agent.stream({"messages": [HumanMessage(content=query)]}, 
                              config=config,
                              stream_mode="values"):
        # Each chunk contains the full state at that point
        latest_message = chunk["messages"][-1]
        if latest_message.content:
            if isinstance(latest_message, HumanMessage):
                # print(f"User: {latest_message.content}")
                pass
            elif isinstance(latest_message, AIMessage):
                print(f"Agent: {latest_message.content}")
        elif latest_message.tool_calls:
            print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")

def main():
    """Main entry point for the Research AI Agent.

    Runs an interactive CLI session where users can query the research agent.
    The agent maintains conversation history across sessions.
    """
    banner()
    agent = create_research_agent()
    config = {"configurable": {"thread_id": "research-session-1"}}

    while True:
        try:
            query = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("\nGoodBye! Happy Researching!")
            break

        try:
            stream_response(agent, query, config)
        except Exception as err:
            print(f"Error: {err}")


main()
