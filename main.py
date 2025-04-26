import streamlit as st
import openai
from dotenv import load_dotenv
import os
from datetime import datetime
import json
from tools import TextTools, DataTools, SearchTools

# Page configuration
st.set_page_config(page_title="AI Chat Assistant", page_icon="💬", layout="wide")

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize tools
@st.cache_resource(show_spinner=False)
def init_tools():
    return {
        "text": TextTools(openai.api_key),
        "data": DataTools(),
        "search": SearchTools(openai.api_key)
    }

tools = init_tools()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

def get_tool_description():
    return {
        "Text Summarization": "Summarize long texts into key points",
        "Sentiment Analysis": "Analyze the emotional tone of text",
        "Language Translation": "Translate text between languages",
        "Data Analysis": "Analyze and visualize data from files",
        "Question Answering": "Get answers from provided context",
        "Keyword Extraction": "Extract key terms from text",
        "Grammar Check": "Check and correct grammar issues"
    }

def display_sidebar():
    st.sidebar.title("💡 Tools")
    tools_desc = get_tool_description()
    
    # Tool selection
    selected_tool = st.sidebar.selectbox(
        "Select a tool",
        ["None"] + list(tools_desc.keys()),
        key="tool_selector"
    )
    
    # Tool description and additional inputs
    if selected_tool != "None":
        st.sidebar.markdown(f"**Description:** {tools_desc[selected_tool]}")
        
        if selected_tool == "Language Translation":
            st.session_state.target_language = st.sidebar.text_input(
                "Target Language:",
                value="Spanish",
                key="translation_lang"
            )
        elif selected_tool == "Question Answering":
            st.session_state.context = st.sidebar.text_area(
                "Context:",
                height=150,
                key="qa_context"
            )
    
    return selected_tool

def process_with_tool(text: str, tool_name: str):
    try:
        if tool_name == "Text Summarization":
            result = tools["text"].summarize(text)
            if result["success"]:
                return f"📝 Summary:\n{result['summary']}"
            
        elif tool_name == "Sentiment Analysis":
            result = tools["text"].analyze_sentiment(text)
            if result["success"]:
                sentiment = result["sentiment"]
                confidence = result["confidence"]
                emoji = "😊" if sentiment == "POSITIVE" else "😢" if sentiment == "NEGATIVE" else "😐"
                return f"{emoji} Sentiment: {sentiment}\nConfidence: {confidence:.2%}"
            
        elif tool_name == "Language Translation":
            target_lang = getattr(st.session_state, 'target_language', 'Spanish')
            result = tools["text"].translate(text, target_lang)
            if result["success"]:
                return f"🌐 Translation ({target_lang}):\n{result['translation']}"
            
        elif tool_name == "Keyword Extraction":
            result = tools["text"].extract_keywords(text)
            if result["success"]:
                return f"🔑 Keywords:\n{', '.join(result['keywords'])}"
            
        elif tool_name == "Grammar Check":
            result = tools["text"].check_grammar(text)
            if result["success"]:
                return f"📝 Grammar Analysis:\n{result['analysis']}"
            
        elif tool_name == "Question Answering":
            context = getattr(st.session_state, 'context', '')
            if context:
                result = tools["search"].question_answering(text, context)
                if result["success"]:
                    return f"❓ Answer:\n{result['answer']}"
            else:
                return "⚠️ Please provide context in the sidebar."
            
        if "error" in locals():
            return f"⚠️ Error: {result['error']}"
            
        return get_assistant_response(text)
            
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

def get_assistant_response(prompt: str, tool: str = None) -> str:
    try:
        messages = [{"role": "system", "content": "You are a helpful AI assistant skilled in various tasks including text analysis, data processing, and answering questions."}]
        
        if tool:
            messages[0]["content"] += f" Currently, you are specifically helping with: {tool}"
        
        messages.append({"role": "user", "content": prompt})
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message["content"]
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

def display_chat_message(role: str, content: str, avatar: str = None):
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)

def main():
    st.title("💬 AI Chat Assistant")
    selected_tool = display_sidebar()
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(
            message["role"],
            message["content"],
            "🧑‍💻" if message["role"] == "user" else "🤖"
        )
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Display user message
        display_chat_message("user", prompt, "🧑‍💻")
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Process with selected tool or get general response
        if selected_tool != "None":
            response = process_with_tool(prompt, selected_tool)
        else:
            response = get_assistant_response(prompt)
            
        # Display assistant response
        display_chat_message("assistant", response, "🤖")
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Footer
    st.markdown(
        "<div style='position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px 0; background-color: #f0f2f6;'>Made by Tisha Choksi</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
