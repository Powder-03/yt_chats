from langchain_core.prompts import PromptTemplate

prompt1 = PromptTemplate(
    input_variables=["youtube_url"],
    template="You are a helpful assistant. I will provide you youtube video url. The URL is: {youtube_url}"

prompt2 = PromptTemplate(
    input_variables=["context" , "question"],
    template=" You are an helpful assistant , Answer the question based on the context below. If the answer is not in the context, say 'I don't know'.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
)