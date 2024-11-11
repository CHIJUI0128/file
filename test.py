# from langchain_community.llms import Ollama
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate

# class MyOutputParser(StrOutputParser):
#     def parse(self, text):
#         return text.replace('Assistant: ', '').strip()

# output_parser = MyOutputParser()

# llm = Ollama(model='llama3.2')
# prompt = ChatPromptTemplate.from_messages([
#     ('user', '{input}'),
# ])

# chain = prompt | llm | output_parser

# input_text = input('>>> ')
# while input_text.lower() != 'bye':
#    print(chain.invoke({'input': input_text}))
#    input_text = input('>>> ')



from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import CharacterTextSplitter

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# 初始化Ollama模型
llm = Ollama(model='llama3.2', callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

# 載入並分割PDF文件
loader = PyPDFLoader("Probe_Card.pdf")
docs = loader.load_and_split()

# 設定文本分割器，chunk_size是分割的大小，chunk_overlap是重疊的部分
text_splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=5)
documents = text_splitter.split_documents(docs)

# 初始化嵌入模型
embeddings = OllamaEmbeddings(model="llama3.2")

# 使用FAISS建立向量資料庫
vectordb = FAISS.from_documents(docs, embeddings)
# 將向量資料庫設為檢索器
retriever = vectordb.as_retriever()

# 設定提示模板，將系統和使用者的提示組合
prompt = ChatPromptTemplate.from_messages([
    ('system', 'Answer the user\'s questions in Chinese, based on the context provided below:\n\n{context}'),
    ('user', 'Question: {input}'),
])

# 創建文件鏈，將llm和提示模板結合
document_chain = create_stuff_documents_chain(llm, prompt)

# 創建檢索鏈，將檢索器和文件鏈結合
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# context = []
input_text = input('>>> ')
while input_text.lower() != 'bye':
    response = retrieval_chain.invoke({
        'input': input_text,
        # 'context': context
    })
    # print(response['answer'])
    # context = response['context']
    input_text = input('>>> ')




from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct


from langchain.vectorstores import Qdrant

from langchain_ollama import OllamaEmbeddings


embeddings = OllamaEmbeddings(model="llama3.2")

text_array = ["我會披星戴月的想你，我會奮不顧身的前進，遠方煙火越來越唏噓，凝視前方身後的距離",
              "而我，在這座城市遺失了你，順便遺失了自己，以為荒唐到底會有捷徑。而我，在這座城市失去了你，輸給慾望高漲的自己，不是你，過分的感情"]

doc_store = Qdrant.from_texts(
    text_array, embeddings, url="http://localhost:6333", collection_name="Lyrics_Langchain")


question = "工程師寫程式"
docs = doc_store.similarity_search_with_score(question)

document, score = docs[0]
print(document.page_content)
print(f"\nScore: {score}")



# from qdrant_client import QdrantClient

# # 初始化 Qdrant 客戶端
# client = QdrantClient(url="http://localhost:6333")

# # 設定要查詢的集合名稱
# collection_name = "PDF_Langchain"

# # 檢索集合中的前 10 筆數據
# result = client.search(
#     collection_name=collection_name,
#     query_vector=[0.0] * 512,  # 使用隨機向量進行測試或設定為具體查詢向量
#     limit=10  # 限制返回的數據數量
# )

# # 輸出結果
# for point in result:
#     print(point)



from langchain_community.llms import Ollama
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.chat_models.azure_openai import AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain

from langchain_ollama import OllamaEmbeddings

def get_embeddings():
    
    return OllamaEmbeddings(model="llama3.2")


def load_and_split_documents(filepaths=["./Probe_Card.pdf", "./SigilloProbeCard.pdf"]):
    documents = []
    
    # 迴圈載入並處理多個 PDF 檔案
    for filepath in filepaths:
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        
        # 將每個文件分割為片段
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = text_splitter.split_documents(docs)
        
        # 合併到主文件列表
        documents.extend(split_docs)
    
    return documents

def get_document_store(docs, embeddings):
    return Qdrant.from_documents(
        docs,
        embeddings,
        url="http://localhost:6333",
        collection_name="PDF_Langchain",
        force_recreate=True
    )

def get_chat_model():

    return Ollama(model='llama3.2')

def ask_question_with_context(qa, question, chat_history):

    query = ""
    result = qa({"question": question, "chat_history": chat_history})
    print("answer:", result["answer"])
    chat_history = [(query, result["answer"])]
    return chat_history


def main():

    embeddings = get_embeddings()
    docs = load_and_split_documents()
    doc_store = get_document_store(docs, embeddings)
    llm = get_chat_model()

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=doc_store.as_retriever(),
        return_source_documents=True,
        verbose=False
    )

    chat_history = []
    while True:
        query = input('you: ')
        if query == 'q':
            break
        chat_history = ask_question_with_context(qa, query, chat_history)


if __name__ == "__main__":
    main()
