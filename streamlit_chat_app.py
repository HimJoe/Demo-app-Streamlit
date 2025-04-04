import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'openai_key' not in st.session_state:
        st.session_state.openai_key = ""
    if 'anthropic_key' not in st.session_state:
        st.session_state.anthropic_key = ""

def create_conversation_chain(model_name, temperature):
    try:
        # Define the prompt template
        prompt = PromptTemplate(
            input_variables=["chat_history", "input"],
            template="""You are Dexter, a chat bot craetd to be friendly and honest.The following is a friendly conversation between a human and an AI. The AI is helpful, creative, clever, and very friendly.

Current conversation:
{chat_history}
Human: {input}
AI:"""
        )
        
        if "gpt" in model_name.lower():
            if not st.session_state.openai_key:
                st.error("Please enter your OpenAI API key!")
                return None
            llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                openai_api_key=st.session_state.openai_key
            )
        else:
            if not st.session_state.anthropic_key:
                st.error("Please enter your Anthropic API key!")
                return None
            llm = ChatAnthropic(
                model=model_name,
                temperature=temperature,
                anthropic_api_key=st.session_state.anthropic_key
            )
        
        # Initialize memory with the correct key
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="input"
        )
        
        # Create the conversation chain with the custom prompt
        return ConversationChain(
            llm=llm,
            memory=memory,
            prompt=prompt,
            verbose=True
        )
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="AI Chat Assistant", page_icon="🤖", layout="wide")
    
    initialize_session_state()
    
    # Sidebar for settings
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # API Keys input
        st.subheader("API Keys")
        openai_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.openai_key)
        anthropic_key = st.text_input("Anthropic API Key", type="password", value=st.session_state.anthropic_key)
        
        if openai_key != st.session_state.openai_key:
            st.session_state.openai_key = openai_key
            st.session_state.conversation = None
        
        if anthropic_key != st.session_state.anthropic_key:
            st.session_state.anthropic_key = anthropic_key
            st.session_state.conversation = None
        
        # Model selection
        st.subheader("Model Settings")
        model_options = {
            "Claude 35 Haiku": "claude-3-5-haiku-20241022",
            "Claude 35 Sonnet": "claude-3-5-sonnet-20241022",
            "GPT-4o-mini": "gpt-4o-mini",
            "GPT-3.5 Turbo": "gpt-3.5-turbo"
        }
        selected_model = st.selectbox("Select Model", list(model_options.keys()))
        model_name = model_options[selected_model]
        
        # Temperature slider
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

        
        # Reset button
        if st.button("Reset Conversation"):
            st.session_state.messages = []
            st.session_state.conversation = None
            st.success("Conversation reset!")
    
    # Main chat interface
    st.title("💬 AI Chat Assistant")
    
    # Initialize conversation if needed
    if st.session_state.conversation is None:
        st.session_state.conversation = create_conversation_chain(model_name, temperature)
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        if st.session_state.conversation is None:
            st.error("Please configure your API keys first!")
            return
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display "thinking" message
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.conversation.predict(input=prompt)
                    st.markdown(response)
                    # Add assistant response to session state
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
