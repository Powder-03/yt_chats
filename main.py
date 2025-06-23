import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import re
from dotenv import load_dotenv

load_dotenv()


st.set_page_config(page_title="YT_chats", layout="centered")

st.title("📺 YT_chats 🤖")
st.markdown("Enter a YouTube URL and ask questions about the video content.")

# Step 1: Input YouTube URL
youtube_url = st.text_input("Enter YouTube video URL")

def extract_video_id(url):
  
    match = re.search(r"(?:v=|youtu\.be/)([^&]+)", url)
    return match.group(1) if match else None

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if youtube_url:
    video_id = extract_video_id(youtube_url)
    if video_id:
        try:
            st.info("Fetching transcript...")
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            transcript = " ".join(chunk["text"] for chunk in transcript_list)

            # Split text
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents([transcript])

            # Generate embeddings and vector store
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.from_documents(chunks, embeddings)
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            st.session_state.vector_store = retriever
            st.success("Transcript indexed successfully! You can now chat about the video.")
        except TranscriptsDisabled:
            st.error("Transcripts are disabled for this video.")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Invalid YouTube URL.")

# Step 2: Chat Interface
if st.session_state.vector_store:
    user_input = st.chat_input("Ask a question about the video...")

    if user_input:
        retriever = st.session_state.vector_store

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Prompt template
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a helpful assistant. Answer the question based on the context below.\n\n"
                "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            )
        )

        # Chain setup
        parallel_chain = RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })

        parser = StrOutputParser()
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0.7)
        main_chain = parallel_chain | prompt | model | parser

        # Invoke the chain
        response = main_chain.invoke(user_input)

        # Update chat history
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", response))

    # Display chat history
    for speaker, message in st.session_state.chat_history:
        with st.chat_message(speaker):
            st.markdown(message)
