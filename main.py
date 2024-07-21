import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain, ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import time
import sys
from htmltemplate import bot_template, user_template , css

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question, 
    and only use the information from the context provided.
    if answer not in the context, say that i doesn't have that information please check on website and
    other sources of information , and if you have any quary please call on 9876543210 and we will get back to you.
    <context>
    {context}
    <context>
    Question:{input}
    """
)

# Initialize session state attributes
if 'vectors' not in st.session_state:
    st.session_state.vectors = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

# Function to load vector embeddings
def load_vector_embeddings():
    if st.session_state.vectors is None:
        try:
            st.session_state.embeddings = OllamaEmbeddings(model="llama3")
            st.session_state.vectors = FAISS.load_local("faiss_index_QA", st.session_state.embeddings, allow_dangerous_deserialization=True)
            st.success("Vector Database loaded successfully")
        except Exception as e:
            st.error(f"Error loading FAISS index: {e}")

# Function to get conversation chain
def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to handle user input
def handle_userinput(user_question):
    if st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                # Use user template
                st.markdown(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                # Use bot template
                st.markdown(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.error("Conversation chain not initialized.")

# Main function
def main():
    st.set_page_config(page_title="Question & Answering chatbot for Trading ", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Aaritya Technology (SAHI)")

    if st.button("ready to ask Quary"):
        load_vector_embeddings()
        if st.session_state.vectors is not None:
            st.session_state.conversation = get_conversation_chain(st.session_state.vectors)

    user_question = st.text_input("Please ask your question to chatbot and click on Enter button:")

    if user_question:
        handle_userinput(user_question)

# Run the main function
if __name__ == "__main__":
    main()
