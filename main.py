# from langchain.llms import GooglePalm
# from langchain_community.llms import GooglePalm
import os
from dotenv import load_dotenv
# from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import  RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA


load_dotenv()
os.environ["GOOGLE_API_KEY"]

# llm=GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"],temperature=0.1)

llm=GoogleGenerativeAI(model="gemini-pro",google_api_key=os.environ["GOOGLE_API_KEY"])

# Store the loaded data in the 'data' variable


instructor_embeddings = HuggingFaceInstructEmbeddings()

vectordb_file_path="faiss_index"
def create_vector_db():
    loader = CSVLoader(file_path='codebasics_faqs.csv', source_column="prompt",encoding='cp1252')
    data = loader.load()
    vectordb=FAISS.from_documents(documents=data,embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():

    vectordb=FAISS.load_local(vectordb_file_path,instructor_embeddings,allow_dangerous_deserialization=True)

    retriever=vectordb.as_retriever(score_thresold=0.7)

    

    prompt_template="""Given the following context and a question,
    generate an answer based on this context only.In the answer 
    try to provide as much text as possible from "response"  section
    in the source document context without making much changes.
    If the answer is not found in the context, kindly state 
    "I dont know".Dont try to makeup an answer.

    CONTEXT:{context}
    QUESTION:{question}
    """
    PROMPT=PromptTemplate(template=prompt_template,input_variables=["context","question"])
    
    chain=RetrievalQA.from_chain_type(llm=llm,
            chain_type="stuff",
            retriever=retriever,
            input_key="query",
            return_source_documents=True,
            chain_type_kwargs={"prompt":PROMPT})
    return chain
    # chain("Tell me about codebasics platform")

# if __name__ =="__main__": 
#     # create_vector_db()
#     chain = get_qa_chain()
#     # print(chain("do you provide internship?"))    