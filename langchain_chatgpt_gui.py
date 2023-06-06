import streamlit as st
from streamlit_chat import message

import huggingface
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
    )




def main():

    st.set_page_config(
        page_title="Your own ChatGPT",
        page_icon=":|"
    )

    chat = ChatOpenAI(temperature=0.3)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant."),
        ]

    st.header("Your own ChatGPT")

    with st.sidebar:
        user_input = st.text_input("Your message ", key="user_input")

        if user_input:
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("Thinking..."):
                response = chat(st.session_state.messages)
            st.session_state.messages.append(AIMessage(content=response.content))
        
    messages = st.session_state.get("messages", [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + "_user")
        else:
            message(msg.content, is_user=False, key=str(i) + "_ai")


if __name__ == "__main__":
    main()
