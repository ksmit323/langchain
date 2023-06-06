from dotenv import find_dotenv, load_dotenv
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

# -------------------------------------------------------------------------
# # Load API key
# load_dotenv()

# chat = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

# messages = [
#     SystemMessage(content='You are a helpful assistant'),
# ]
# print("Hello, I am ChatGPT CLI")

# while True:
#     user_input = input("> ")

#     messages.append(HumanMessage(content=user_input))

#     ai_response = chat(messages)

#     messages.append(AIMessage(content=ai_response.content))

    # print("\nAssistant: Hello, how can I assist you?")
# -------------------------------------------------------------------------

llm = ChatOpenAI(temperature=0.7)
conversation = ConversationChain(
                llm=llm, 
                memory=ConversationEntityMemory(llm=llm),
                prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE, 
                verbose=False
                )

while True:
    user_input = ("> ")

    ai_response = conversation.predict(input=user_input)

    print("\nAssistant: Hello, how can I assist you?")

