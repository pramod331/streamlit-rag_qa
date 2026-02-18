import os

import streamlit as st
import certifi
# print(certifi.where())


from rag_utility import process_document_to_chroma_db, answer_question

# set the working directory
# working_dir = os.getcwd()
working_dir = os.path.dirname(os.path.abspath((__file__)))

st.title(" Llama-3.3-70B - Document RAG - Pramod")

#file uploader widget
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    #define save path
    save_path = os.path.join(working_dir,uploaded_file.name)
    # save the file
    with open(save_path,"wb") as f:
        f.write(uploaded_file.getbuffer())

    # process_document = process_document_to_chroma_db(uploaded_file.name)
    with st.spinner("Processing document..."):
    process_document_to_chroma_db(uploaded_file.name)


    st.info("Document Processed Successfully")

#text widget to get user input
user_question = st.text_area("Ask your question about the document")

if st.button("Answer"):

    # answer = answer_question(user_question)
    with st.spinner("Thinking..."):
    answer = answer_question(user_question)


    st.markdown("###  Llama-3.3-70B Response")

    st.markdown(answer)

