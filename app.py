import os
import streamlit as st
import certifi
import sys

# Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION environment variable
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from rag_utility import process_document_to_chroma_db, answer_question

# Set the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Create a Streamlit app
st.title("Document RAG Application - Pramod")

# File uploader widget
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_file = os.path.join(working_dir, "temp.pdf")
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Check if the file exists
    if os.path.exists(temp_file):
        with st.spinner("Processing document..."):
            process_document_to_chroma_db(temp_file)
        st.success("Document processed!")
    else:
        st.error("File not found")
# if uploaded_file is not None:
#     # Save the uploaded file to a temporary location
#     temp_file = os.path.join(working_dir, "temp.pdf")
#     with open(temp_file, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     try:
#         # Process the document
#         with st.spinner("Processing document..."):
#             process_document_to_chroma_db(temp_file)

#         st.success("Document processed!")
#     except Exception as e:
#         st.error(f"Error processing document: {e}")

# Text widget to get user input
user_question = st.text_area("Ask your question about the document")

if st.button("Answer") and user_question:
    try:
        # Answer the user's question
        with st.spinner("Thinking..."):
            answer = answer_question(user_question)

        st.markdown(answer)
    except Exception as e:
        st.error(f"Error answering question: {e}")
