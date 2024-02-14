# Loading and processing text files
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Agent configuration
from langchain.tools.retriever import create_retriever_tool
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.prompts import MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor

# App creation
import streamlit as st
import time

# API key
# OPEN_AI_KEY = YOUR PERSONAL OpenAI API KEY

def load_txt_files(directory : str):

    '''This function takes in input a directory path and then loads every text files in it.
    Next step is splitting the documents in small pieces and with an OpenAI model it returns
    a vector-based database with the docs and their vectorial representation'''

    # Load the text files in the directory "data"
    loader = DirectoryLoader(directory, glob="*.txt")
    documents = loader.load()

    # Split the text files content into smaller chunks
    markdown_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    docs = markdown_splitter.split_documents(documents)

    # Initialize OpenAI embedding model
    embeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_KEY)

    # Convert all chunks into vectors embeddings using OpenAI embedding model and storing them in Chroma index
    db = Chroma.from_documents(docs, embeddings)
    return db

dir_path = './data/'

# Configure Streamlit page
st.set_page_config(page_title="Your Legal Chatbot", page_icon='👨‍💻')
st.title('Legal Conversation')

# Loading the text files from the directory data
db = load_txt_files(dir_path)

# Load our Chroma database as a retriever
retriever = db.as_retriever()

# Defining the retriever tool for the agent
tool = create_retriever_tool(retriever,
                             name = "Lawyer",
                             description = "Searching through legal documents")

tools = [tool]

# Initialize the memory
if "memory" not in st.session_state.keys():
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the chat message history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello, how can I be of assistance today?"}]

# Chatbot behavior
system_message = SystemMessage(
        content=("""You are a legal AI assistant for answering questions about the inheritance and divorce laws in Italy.
            You are given parts of the italian laws and a question. Provide a professional answer and include the number of articles you are using to answer.
            If you don't know the answer, just say "Sorry, I don't know ...". Don't try to make up an answer.
            If the question is not about the topics of the documents, politely inform them that you are tuned to only answer 
            questions about inheritance and divorce laws in Italy.""")
    )

# Defining the prompt template
prompt_template = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name='chat_history')]
    )

# Load OpenAI chat model (deterministic)
llm = ChatOpenAI(temperature = 0, openai_api_key=OPEN_AI_KEY)

# Create the agent with the retrieval tool
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt_template)
agent_executor = AgentExecutor(agent=agent, tools = tools, memory=st.session_state.memory, verbose=True)

with st.sidebar:

    st.title('ℹ️ More info')
    st.info(f""" This chatbot is custom trained on italian laws about inheritance and divorces.
                 It was created using the LangChain python library (https://www.langchain.com/).
                 The app was developed with the Streamlit python library (https://streamlit.io/)""")


# Print again all the saved chat history if there's a rerun of the streamlit app
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat stream
if query := st.chat_input("Ask me something"):

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})

    # Print user message in the chat message box
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            message_placeholder = st.empty()

            # Query user's question to the agent
            result = agent_executor.invoke({"input": query})
            response = result['output']

        full_response = ""

        # Simulate typing from chatbot
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "|")

        message_placeholder.markdown(response)

    # Append chatbot message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
