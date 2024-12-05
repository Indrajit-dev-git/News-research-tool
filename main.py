import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import UnstructuredURLLoader
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import time

load_dotenv()


st.title("News Research Tool 📈")
st.sidebar.title("New Article URLs")

urls = []

for i in range(3):
    url=st.sidebar.text_input(f"URL{i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button('Process URLs')

embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
llm = GoogleGenerativeAI(model='gemini-1.5-pro')

main_placeholder = st.empty()
if process_url_clicked:

    #load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    #split data
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n','\n','.',','],chunk_size=1000,chunk_overlap=100)
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    #create embeddings and save it to FAISS index
    vectorindex = FAISS.from_documents(docs,embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    #Save the FAISS index
    vectorindex.save_local("faiss_index")

query = main_placeholder.text_input('Question: ')
if query:
    if os.path.exists('faiss_index'):
        loaded_vectorindex = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=loaded_vectorindex.as_retriever())
        result = chain({'question':query},return_only_outputs=True)
        st.header('Answer')
        st.write(result['answer'])

        #Display sources, if available
        sources = result.get('sources','')
        if sources:
            st.subheader('Sources: ')
            sources_list = sources.split('\n')
            for source in sources_list:
                st.write(source)