import streamlit as st
import asyncio
from index_helper import create_query_engine, create_docs_and_nodes, create_vector_store

st.set_page_config(page_title="CodeChat", layout="centered")

st.title("🦙 CodeChat: Talk with your GitHub Repos 💬 📚")

# Function to initialize the query engine
async def init_engine(github_url):
    docs, nodes = create_docs_and_nodes(github_url)
    # docs, nodes = create_docs_and_nodes('https://github.com/carolinedlu/llamaindex-chat-with-streamlit-docs')
    vector_store = create_vector_store(docs, nodes)
    query_engine = await create_query_engine(vector_store, nodes)
    return query_engine

with st.form('my_form'):
    github_url = st.text_area('Enter text:',  placeholder='Eg: https://github.com/zmusaddique/chatbot-restaurant')
    submitted = st.form_submit_button('Submit')
    # github_url = 'https://github.com/zmusaddique/chatbot-restaurant'


query_engine = asyncio.run(init_engine(github_url)) if github_url and submitted else print('error')
query_engine = asyncio.run(init_engine(github_url)) if github_url else print('error')

if query_engine:

    # Initialize the chat message history
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question about your GitHub Repository!"}
        ]

    # User input for the chat
    if user_input := st.chat_input("Your question", key="user_input"):
        st.session_state.messages.append({"role": "user", "content": user_input})


    # Display chat messages
    for message in st.session_state.messages:   
        with st.chat_message(message['role']):
            with st.empty():
                st.write(message["content"])

    # If the user input is not from the assistant, query the engine and display the response
    if st.session_state.messages[-1]["role"] != "assistant" and user_input:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = query_engine.query(user_input)
                st.write(response.response)
                st.session_state.messages.append({"role": "assistant", "content": response.response})
