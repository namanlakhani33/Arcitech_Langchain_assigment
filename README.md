
# Arcitech Rag Chatbot Using Langchain and OpenAI

This project is a chatbot built using Langchain and OpenAI, designed to query information from uploaded PDF documents and provide responses with a typewriter effect in the Streamlit application.


## Run Locally

Clone the project

```bash
  git clone https://github.com/namanlakhani33/Arcitech_Langchain_assigment
```

Go to the project directory

```bash
  cd LangChain_assignment
```

 Create a Virtual Environment
```bash
 python -m venv venv

```
Activate the Virtual Environment
```bash
 venv\Scripts\activate

```
Install Requirements
```bash
 pip install -r requirements.txt
```
Install Requirements
```bash
streamlit run app4.py
```

![Screenshot 2024-06-22 122100](https://github.com/namanlakhani33/Arcitech_Langchain_assigment/assets/97312875/c3dac24a-b8b3-4fd8-9934-bd9c50fc863b)
![Screenshot 2024-06-22 122242](https://github.com/namanlakhani33/Arcitech_Langchain_assigment/assets/97312875/2b47dc34-6c4a-47d3-a2a8-4693acafd4bc)
![Screenshot 2024-06-22 122302](https://github.com/namanlakhani33/Arcitech_Langchain_assigment/assets/97312875/0a1687d7-fde1-4491-996e-c93620909015)

# Flow

#### The user first sets their OpenAI API key and uploads a PDF file. The PDFProcessingAgent processes the PDF by loading its content and creating vector embeddings using Chroma. These embeddings are stored in a vector database, allowing efficient search and retrieval of document information. When the user submits a query, the QueryHandlingAgent interacts with the ConversationalRetrievalChain, which uses the embeddings to retrieve relevant information from the processed PDF. This information is then used to generate a response with OpenAI's language model. The response is displayed with a typewriter effect, and the relevant page from the PDF is rendered and shown to the user.



# Application Overview
## Features

#### Upload PDF:
Allows users to upload a PDF file for querying.
#### Query Handling: 
Accepts user queries and retrieves relevant information from the uploaded PDF.
#### Typewriter Effect: 
Displays responses with a typewriter effect for a dynamic user experience.
PDF 
#### Page Rendering: 
Renders and displays the relevant page from the PDF based on the query.

## How It Works

#### Set OpenAI API Key: 
Enter your OpenAI API key in the text input field and click "Set API Key" to set the API key for the session.
#### Upload PDF:
Upload a PDF file using the file uploader. The application will process the PDF and prepare it for querying.
#### Enter Query: 
Input your query in the text box and click "Submit Query".
#### View Response:
The chatbot will display the response with a typewriter effect. Additionally, the relevant page from the PDF will be rendered and displayed.


## License

[MIT](https://choosealicense.com/licenses/mit/)


## Example Usage

Enter your OpenAI API key:

Upload a PDF file:

Enter your query:

View the response and rendered PDF page:

