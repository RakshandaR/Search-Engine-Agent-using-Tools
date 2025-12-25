# showcasing which tool calls have been made

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

st.title("🧠 LangChain Agent - Tools & Reasoning")

# Sidebar
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password", value=groq_api_key or "")

if not api_key:
    st.info("Please add your Groq API key in the sidebar.")
    st.stop()

# SIMPLIFIED WORKING TOOLS
@tool
def arxiv_search(query: str) -> str:
    """Search Arxiv for academic papers."""
    from langchain_community.utilities import ArxivAPIWrapper
    from langchain_community.tools import ArxivQueryRun
    wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    return ArxivQueryRun(api_wrapper=wrapper).run(query)

@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for general knowledge."""
    from langchain_community.utilities import WikipediaAPIWrapper
    from langchain_community.tools import WikipediaQueryRun
    wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    return WikipediaQueryRun(api_wrapper=wrapper).run(query)

search = DuckDuckGoSearchRun(name="web_search")
tools = [search, arxiv_search, wikipedia_search]

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me anything and see my tools & reasoning."}]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input - RELIABLE INVOKE + TOOL DISPLAY
if prompt := st.chat_input("Try: 'What is ML?' or 'Latest AI papers'..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Create agent
        if "agent" not in st.session_state:
            llm = ChatGroq(
                groq_api_key=api_key,
                model="llama-3.1-8b-instant",
                temperature=0,
                max_tokens=1500
            )
            st.session_state.agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt="""You are a helpful assistant. 

TOOLS:
• web_search: Current info, news
• wikipedia_search: Definitions, facts  
• arxiv_search: Research papers

Use tools when needed. Respond concisely with final answer."""
            )

        agent = st.session_state.agent
        
        # Convert messages
        messages = []
        for m in st.session_state.messages[:-1]:
            if m["role"] == "user":
                messages.append(HumanMessage(content=m["content"]))
            else:
                messages.append(AIMessage(content=m["content"]))
        messages.append(HumanMessage(content=prompt))
        
        # RELIABLE INVOKE
        with st.spinner("Thinking..."):
            result = agent.invoke({"messages": messages})
        
        # EXTRACT FULL REASONING + TOOLS
        all_messages = result["messages"]
        final_response = all_messages[-1].content
        
        # DISPLAY TOOL USAGE & REASONING
        st.markdown("### 🔍 **Tools Used & Reasoning**")
        
        tool_usage = []
        for i, msg in enumerate(all_messages):
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    tool_usage.append(f"🔧 **{tool_name}**(`{tool_args}`)")
            
            if isinstance(msg, AIMessage) and msg.content and "tool_calls" not in str(type(msg)):
                st.markdown(f"**🤔 Thought:** {msg.content}")
                st.markdown("---")
        
        if tool_usage:
            st.markdown("### 🛠️ **Tool Calls:**")
            for tool_call in tool_usage:
                st.markdown(tool_call)
            st.markdown("---")
        
        # FINAL ANSWER
        st.markdown("### ✅ **Final Answer**")
        st.markdown(final_response)
        
        # Save to session
        st.session_state.messages.append({"role": "assistant", "content": f"**Tools Used:** {', '.join(tool_usage)}\n\n**Answer:** {final_response}"})