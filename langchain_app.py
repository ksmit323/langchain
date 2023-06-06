import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

apikey = ...
os.environ['OPEN_AI_KEY'] = apikey

# App framework
st.title('🦜🔗 LangChain')
prompt = st.text_input('Plug in your prompt here')

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'Write me a title about {topic}'
)
script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'],
    template = 'Write me a script based on this title TITLE: {title} while leveraging wikipedia {wikipedia_research}'
)

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# LLMs
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

wiki = WikipediaAPIWrapper()

# Show to the screen
if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    with st.expander('Message History'):
        st.info(title_memory.buffer)

    with st.expander('Message History'):
        st.info(script_memory.buffer)

    with st.expander('Message History'):
        st.info(wiki_research.buffer)

