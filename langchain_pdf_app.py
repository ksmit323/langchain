from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

# Command to run streamlit: streamlit run <app>.py

def main():
    # load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF")

    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # Extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        

        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size= 1000,
            chunk_overlap = 200,
            length_function = len
        )
        chunks = text_splitter.split_text(text)

        # Convert chunks into embeddings
        embeddings = OpenAIEmbeddings(openai_api_key="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")

        # Create knowledge base
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Prompt user for question
        user_query = st.text_input("Ask a question about your PDF:")

        # Create response from user query
        if user_query:
            docs = knowledge_base.similarity_search(user_query)
            llm = OpenAI(openai_api_key="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")
            chain = load_qa_chain(llm, chain_type="stuff")

            # Track spending amount with callback 
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_query)
                print(cb)


            # Generate response from LLM
            st.write(response)
        


if __name__ == '__main__':
    main()