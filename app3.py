import os
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
import fitz
import streamlit as st
from PIL import Image

# Global variables
COUNT = 0
chat_history = []
chain = ''
N = 0

def set_apikey(api_key):
    os.environ['OPENAI_API_KEY'] = api_key
    print("API key set successfully.")

def process_file(file_path):
    # Raise an error if API key is not provided
    if 'OPENAI_API_KEY' not in os.environ:
        raise ValueError('Upload your OpenAI API key')
    
    # Load the PDF file using PyPDFLoader
    loader = PyPDFLoader(file_path) 
    documents = loader.load()
    
    # Initialize OpenAIEmbeddings for text embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create a ConversationalRetrievalChain with ChatOpenAI language model
    # and PDF search retriever
    pdfsearch = Chroma.from_documents(documents, embeddings)
    
    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.3), 
        retriever=pdfsearch.as_retriever(search_kwargs={"k": 1}),
        return_source_documents=True
    )
    return chain

def generate_response(query, file_path):
    global COUNT, chat_history, chain, N
    
    # Check if a PDF file is uploaded
    if not file_path:
        raise ValueError('Upload a PDF')
    
    # Initialize the conversation chain only once
    if COUNT == 0:
        chain = process_file(file_path)
        COUNT += 1
    
    # Generate a response using the conversation chain
    result = chain({"question": query, 'chat_history': chat_history}, return_only_outputs=True)
    
    # Update the chat history with the query and its corresponding answer
    chat_history.append((query, result["answer"]))
    
    # Retrieve the page number from the source document
    N = list(result['source_documents'][0])[1][1]['page']
    
    return result["answer"]

def render_file(file_path):
    global N
    
    # Open the PDF document using fitz
    doc = fitz.open(file_path)
    
    # Get the specific page to render
    page = doc[N]
    
    # Render the page as a PNG image with a resolution of 300 DPI
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    
    # Save the rendered image to a file
    image_path = "rendered_page.png"
    pix.save(image_path)
    
    print(f"Rendered image saved to {image_path}")

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

    query = st.text_input("Enter your query:")
    
    if st.button("Submit Query"):
        if 'file_path' in st.session_state:
            response = generate_response(query, st.session_state['file_path'])
            st.write("Response:", response)
            
            image_path = render_file(st.session_state['file_path'])
            image = Image.open(image_path)
            st.image(image, caption='Rendered PDF Page', use_column_width=True)
        else:
            st.error("Please upload a PDF file first.")

if __name__ == "__main__":
    main()
