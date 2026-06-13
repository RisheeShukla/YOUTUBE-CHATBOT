from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace,HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import streamlit as st
load_dotenv()
import os
HUGGINGFACEHUB_API_TOKEN=os.getenv('HUGGINGFACE_API_TOKEN')
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-235B-A22B-Instruct-2507",
    task="text-generation",
   huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    temperature=2,
    max_new_tokens=100
)
model=ChatHuggingFace(llm=llm)


st.title("Youtube Chatbot")

st.header("Enter Video id")
video_id = st.text_input("Video Id")
if(video_id!=""):
   st.video(f"https://www.youtube.com/watch?v={video_id}")

st.header("Enter your question or query")
query = st.text_input("Query")


def fetch_info():
   try:
     api = YouTubeTranscriptApi()
     transcript_list = api.fetch(video_id,languages=['en'])
     transcript = " ".join(entry.text for entry in transcript_list)
   except Exception as e:
      st.chat_message("assistant").write(f"No captions available for this video {str(e)}")
      return None
   

   text_splitter=RecursiveCharacterTextSplitter(
   chunk_size=500,
   chunk_overlap=40
)
   chunks=text_splitter.create_documents([transcript])
   embedding_model=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
   )
   vector_store=Chroma.from_documents(chunks,embedding_model)
   retriever=vector_store.as_retriever(search_tye='mmr',search_kwargs={"k":5})
   def format_docs(retrieved_docs):
     context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
     return context_text


   parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
  })

   template = """
   You are a helpful assistant.
   Use only the transcript context to answer. 
   If no answer possible then generate "Sorry I don't know"

   Context:
    {context}

   Question:
   {question}

   Answer:
   """

   prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
   )

   chain = (
    parallel_chain
    | prompt
    | model
    | StrOutputParser()
 )

   answer = chain.invoke(query)
   st.chat_message("assistant").write(answer)
 
if query!="" and video_id!="":
  st.button("Save",on_click=fetch_info)

   
