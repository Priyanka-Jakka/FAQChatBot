import streamlit as st
from main import create_vector_db,get_qa_chain


st.title("CODEBASICS QA ")
btn=st.button("create knowledge")
if btn:
    pass


question=st.text_input("Question:")

if question:
    chain=get_qa_chain()
    response = chain(question)

    st.header("Answer:")
    st.write(response["result"])
    