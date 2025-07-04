﻿# 📺 YT_chats 🤖

https://github.com/user-attachments/assets/889b4d98-8fce-43b9-8ea6-8daef49aa2ed



A Streamlit-based application that allows you to chat with YouTube video content using AI. Simply provide a YouTube URL, and the app will extract the video transcript and enable you to ask questions about the video content using Google's Generative AI.

## ✨ Features

- **YouTube Transcript Extraction**: Automatically fetches transcripts from YouTube videos
- **AI-Powered Q&A**: Ask questions about video content and get intelligent responses
- **Vector Search**: Uses FAISS for efficient similarity search through video content
- **Interactive Chat Interface**: Built with Streamlit for a user-friendly experience
- **Real-time Processing**: Processes videos and answers questions in real-time

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd yt_chats
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory and add your Google API key:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## 🔧 Usage

1. **Run the application**
   ```bash
   streamlit run main.py
   ```

2. **Open your browser** and navigate to the provided localhost URL (typically `http://localhost:8501`)

3. **Enter a YouTube URL** in the input field

4. **Wait for processing** - the app will extract and index the video transcript

5. **Start chatting** - ask questions about the video content in the chat interface

## 📋 Requirements

- Python 3.7+
- Google API key for Generative AI
- Internet connection for YouTube transcript fetching

## 🛠️ Dependencies

- `streamlit` - Web application framework
- `youtube_transcript_api` - YouTube transcript extraction
- `langchain` - LLM application framework
- `langchain-google-genai` - Google Generative AI integration
- `faiss-cpu` - Vector similarity search
- `python-dotenv` - Environment variable management

See [`requirements.txt`](requirements.txt) for the complete list.

## 📁 Project Structure

```
yt_chats/
├── main.py           # Main Streamlit application
├── prompts.py        # Prompt templates
├── requirements.txt  # Python dependencies
├── .env             # Environment variables (create this)
├── .gitignore       # Git ignore rules
└── README.md        # Project documentation
```

## 🔍 How It Works

1. **URL Processing**: Extracts video ID from YouTube URL
2. **Transcript Fetching**: Uses YouTube Transcript API to get video transcript
3. **Text Chunking**: Splits transcript into manageable chunks using RecursiveCharacterTextSplitter
4. **Embedding Generation**: Creates embeddings using Google's embedding model
5. **Vector Storage**: Stores embeddings in FAISS for efficient retrieval
6. **Question Processing**: Uses RAG (Retrieval-Augmented Generation) to answer questions
7. **Response Generation**: Generates responses using Google's Gemini model

## ⚠️ Limitations

- Only works with videos that have English transcripts enabled
- Requires a valid Google API key for Generative AI
- May not work with private or restricted YouTube videos

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🔗 Links

- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [Google AI Documentation](https://ai.google.dev/)
