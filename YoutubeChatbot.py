from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace,HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-235B-A22B-Instruct-2507",
    task="text-generation",
   
    temperature=2,
    max_new_tokens=100
)
model=ChatHuggingFace(llm=llm)
# only use youtube id not the whole URL
video_id="YBj6yS6XPzU"
try:
    api = YouTubeTranscriptApi()
    transcript_list = api.fetch(video_id,languages=['en'])

    transcript = " ".join(entry.text for entry in transcript_list)
    print(transcript)
except TranscriptsDisabled:
    print("No captions available for this video")

text_splitter=RecursiveCharacterTextSplitter(
   chunk_size=500,
   chunk_overlap=40
)
chunks=text_splitter.create_documents([transcript])
embedding_model=HuggingFaceEmbeddings()
vector_store=Chroma.from_documents(chunks,embedding_model)
retriever=vector_store.as_retriever(search_tye='mmr',search_kwargs={"k":5})


prompt=PromptTemplate(
    template="""  You are a helpful assistant.
      Answer in numerical points in different lines only from the provided transcript context.
    If the context is insufficient,just say don't know.
    {context}
    Question:{question}
    
    """,
    input_variables=['context','question']

)
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text


parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

result=parallel_chain.invoke('Will Putin stop bombing Ukraine')
print(result)
