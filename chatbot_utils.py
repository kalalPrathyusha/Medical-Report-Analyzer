# chatbot_utils.py

import os
from dotenv import load_dotenv

from langchain_community.llms import Together
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain_core.retrievers import BaseRetriever
from typing import List
from pydantic import Field

# ✅ Load API key from .env file
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# ✅ Define a retriever using BaseRetriever
class CombinedSummaryRetriever(BaseRetriever):
    combined: str = Field(...)  # ✅ Properly declared Pydantic field

    def get_relevant_documents(self, query: str) -> List[Document]:
        return [Document(page_content=self.combined)]

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)
    
# ✅ Main function to return chatbot chain
def get_langchain_chatbot(text_summary: str, image_summary: str):
    combined = f"Text Summary:\n{text_summary}\n\nImage Summary:\n{image_summary}"

    llm = Together(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        temperature=0.3,
        max_tokens=512,
        top_p=0.9,
        together_api_key=TOGETHER_API_KEY  
    )
    retriever = CombinedSummaryRetriever(combined=combined)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False
    )

    return qa_chain


