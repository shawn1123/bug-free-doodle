# agent.py

import asyncio
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

async def main():
    client = MultiServerMCPClient({
        "math": {
            "command": "python",
            "args": ["math_server.py"],  # path to your MCP server
            "transport": "stdio",
        }
    })
    tools = await client.get_tools()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    agent = create_react_agent(llm, tools)

    response = await agent.ainvoke({
        "messages": [{"role": "user", "content": "What is (3 + 5) * 12?"}]
    })
    print(response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
