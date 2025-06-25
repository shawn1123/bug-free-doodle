# agent.py

import os, asyncio
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

async def main():
    # Connect to your remote MCP server
    url = "https://<your-domain>/mcp"
    headers = {"Authorization": f"Bearer {os.getenv('MCP_TOKEN')}"}
    async with streamablehttp_client(url=url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)      # imports add/multiply/divide

            llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY)
            agent = create_react_agent(llm, tools)

            resp = await agent.ainvoke({"messages": [{"role": "user", "content": "What is 10 times 5 minus 2?"}]})
            print(resp["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
