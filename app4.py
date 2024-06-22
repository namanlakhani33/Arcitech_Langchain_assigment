import os
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
import fitz
import streamlit as st
from PIL import Image
import time

# Global variables
COUNT = 0
chat_history = []
N = 0

# Define PDF Processing Agent
class PDFProcessingAgent:
    def __init__(self):
        pass

    def run(self, file_path):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        embeddings = OpenAIEmbeddings()
        pdfsearch = Chroma.from_documents(documents, embeddings)
        chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0.3),
            retriever=pdfsearch.as_retriever(search_kwargs={"k": 1}),
            return_source_documents=True
        )
        return chain

# Define Query Handling Agent
class QueryHandlingAgent:
    def __init__(self, chain):
        self.chain = chain

    def run(self, query):
        global chat_history, N
        result = self.chain({"question": query, 'chat_history': chat_history}, return_only_outputs=True)
        chat_history.append((query, result["answer"]))
        N = list(result['source_documents'][0])[1][1]['page']
        return result["answer"], N

# Function to set API key
def set_apikey(api_key):
    os.environ['OPENAI_API_KEY'] = api_key
    st.success("API key set successfully.")

# Function to render the PDF page as an image
def render_file(file_path, page_number):
    doc = fitz.open(file_path)
    page = doc[page_number]
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    image_path = "rendered_page.png"
    pix.save(image_path)
    return image_path

# Function to display text with a typewriter effect
def display_text_with_effect(container, text):
    container.empty()
    full_text = ""
    for char in text:
        full_text += char
        container.write(full_text)
        time.sleep(0.01)

# Main function
def main():
    st.title("PDF Query with OpenAI and LangChain")

    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if st.button("Set API Key"):
        set_apikey(api_key)

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file:
        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state['file_path'] = "uploaded_file.pdf"
        # Initialize PDF Processing Agent
        pdf_agent = PDFProcessingAgent()
        st.session_state['chain'] = pdf_agent.run(st.session_state['file_path'])

    query = st.text_input("Enter your query:")

    if st.button("Submit Query"):
        if 'file_path' in st.session_state and 'chain' in st.session_state:
            query_agent = QueryHandlingAgent(st.session_state['chain'])
            response, page_number = query_agent.run(query)

            response_container = st.empty()
            display_text_with_effect(response_container, response)

            image_path = render_file(st.session_state['file_path'], page_number)
            image = Image.open(image_path)
            st.image(image, caption='Rendered PDF Page', use_column_width=True)
        else:
            st.error("Please upload a PDF file first.")

if __name__ == "__main__":
    main()
