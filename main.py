from youtube_transcript_api import YouTubeTranscriptApi , TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings , ChatGoogleGenerativeAI

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel , RunnableLambda , RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Indexing (Document ingestion)

video_id = 'Gfr50f6ZBvo'  #only the id not full url
try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id , languages=['en']) 
    
    #flatten it to plain text
    
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    
    
except TranscriptsDisabled :
    print("Transcripts are disabled for this video.")
    
# Indexing (text splitting)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.create_documents([transcript])
    
# Indexing (embedding generation and storing in vector store)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_documents(
    chunks,
    embeddings
)


# Retrieval (searching for relevant chunks)
retriever = vector_store.as_retriever(search_type = "similarity" ,search_kwargs={"k": 4})

# Augmentation
model= ChatGoogleGenerativeAI(model = 'gemini-2.5-flash-preview-04-17', temperature=0.7 )

prompt = PromptTemplate(
    input_variables=["context" , "question"],
    template=" You are an helpful assistant , Answer the question based on the context below. If the answer is not in the context, say 'I don't know'.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
)
question          = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs    = retriever.invoke(question)

def format_docs(retrived_docs):
    context_text = "\n\n".join([doc.page_content for doc in retrived_docs])
    return context_text



# Build a chain

parallel_chain = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
   
)

parser = StrOutputParser()

main_chain = parallel_chain | prompt | model | parser

result = main_chain.invoke('can u summarise the video')

print(result)


