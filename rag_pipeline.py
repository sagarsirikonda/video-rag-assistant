# rag_pipeline.py

import os
import re
from dotenv import load_dotenv

# --- LLM and LangChain Imports ---
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

# --- Function to parse the LLM response ---
def parse_llm_response(response: str) -> tuple[str, list[str]]:
    """
    Parses the LLM's response using a two-step cleaning process that
    handles multiple timestamp formats and cleans up trailing characters.
    """
    # Pattern to find individual start-end time pairs (e.g., "00:00:00-00:00:49")
    time_pair_pattern = r'\d{2}:\d{2}:\d{2}-\d{2}:\d{2}:\d{2}'
    
    # 1. Extract all raw time pairs
    found_pairs = re.findall(time_pair_pattern, response)
    timestamps = [f"[{pair}]" for pair in found_pairs]
    
    match = re.search(time_pair_pattern, response)
    if match:
        first_timestamp_start_index = match.start()
        last_bracket_before_ts = response.rfind('[', 0, first_timestamp_start_index)
        
        if last_bracket_before_ts != -1:
            preliminary_answer = response[:last_bracket_before_ts].strip()
        else:
            preliminary_answer = response[:first_timestamp_start_index].strip()
    else:
        preliminary_answer = response.strip()
        
    clean_answer = re.sub(r'(,\s*)+$', '', preliminary_answer).strip()
        
    return clean_answer, timestamps
    

# --- Core RAG Chain Creation ---
def create_rag_chain(vector_store_path: str, llm):
    """
    Creates the complete RAG chain, compatible with the modern vector store.
    """
    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    
    vectorstore = FAISS.load_local(
        vector_store_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    
    def format_docs(docs):
        """
        Formats the retrieved documents to include text and timestamp for the LLM.
        """
        formatted_context = []
        for doc in docs:
            timestamp = doc.metadata.get('timestamp', 'No timestamp available')
            context_line = f"Context: {doc.page_content}\nTimestamp: {timestamp}"
            formatted_context.append(context_line)
        return "\n\n".join(formatted_context)

    # --- Prompt Template (Unchanged) ---
    prompt_template = """
    You are an expert assistant for summarizing and answering questions about video content.
    Your goal is to provide a concise answer to the user's question based ONLY on the provided context from the video transcript.
    Do not use any external knowledge.
    
    At the very end of your answer, you MUST include the timestamps of the relevant context segments.
    Format the timestamps exactly as they appear in the context, like this: [hh:mm:ss-hh:mm:ss].
    If multiple context segments are relevant, include all their timestamps.
    Do not add any labels or introductory text like "Timestamps:" before them; just append the raw [hh:mm:ss-hh:mm:ss] values.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # --- RAG Chain ---
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# --- Testing Block ---
if __name__ == '__main__':
    load_dotenv()

    # --- LLM Configuration with Options ---
    # GROQ (uncomment to use)
    # llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

    # GEMINI
    ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                                temperature=0,
                                google_api_key=os.getenv("GEMINI_API_KEY")
                            )
    
    # OPENAI (uncomment to use)
    # llm = ChatOpenAI(model="gpt-3.5-turbo", 
    #                  temperature=0, 
    #                  api_key=os.getenv("OPENAI_API_KEY")
    #                  )

    # --- Vector Store Configuration for Testing ---
    TEST_VECTOR_STORE_PATH = "data/vector_store/The ginormous collision that tilted our planet - Elise Cutts [vCbx5jtZ_qI]"
    
    if not os.path.exists(TEST_VECTOR_STORE_PATH):
        print("="*50)
        print(f"TESTING SKIPPED: Please update 'TEST_VECTOR_STORE_PATH' in rag_pipeline.py")
        print(f"Current path is: {TEST_VECTOR_STORE_PATH}")
        print("="*50)
    else:
        print("Creating RAG chain...")
        rag_chain = create_rag_chain(TEST_VECTOR_STORE_PATH, llm)
        print("RAG chain created successfully!")
        print("-" * 50)
        
        question = "What is the main topic discussed in the video?"
        print(f"Question: {question}")
        response = rag_chain.invoke(question)
        answer, timestamps = parse_llm_response(response)
        
        print("-" * 50)
        print(f"Clean Answer: {answer}")
        print(f"Retrieved Timestamps: {timestamps}")
        print("-" * 50)