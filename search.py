# import
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.vector_stores.elasticsearch import ElasticsearchStore, AsyncDenseVectorStrategy

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

import requests

# define embedding function
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

es = ElasticsearchStore(
    index_name = "hlm",
    es_url = "http://localhost:9200",
    retrieval_strategy = AsyncDenseVectorStrategy()
)

def print_results(results):
    for rank, result in enumerate(results, start=1):
        title = result.metadata.get("title")
        score = result.get_score()
        text = result.get_text()
        print(f"{rank}. title={title} \nscore={score} \ncontent={text}")

def deepseek_api(qa_prompt:str):
    # 本地ollama部署deepseek-r1
    api_url = "http://localhost:11434/api/generate"

    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek-r1:latest",  # 你的 DeepSeek 模型名称
        "prompt": qa_prompt,
        "stream": False  # 关闭流式输出
    }

    response = requests.post(api_url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
    else:
        print("请求失败:", response.status_code, response.text)
    
    return result["response"]

def search(query: str):
    index = VectorStoreIndex.from_vector_store(vector_store=es)
    retriever = index.as_retriever(similarity_top_k=10)
    results = retriever.retrieve(QueryBundle(query_str=query))
    print_results(results)
    
    context_str = "\n\n".join([n.node.get_content() for n in results])
    qa_prompt = '''你是一个有帮助且知识渊博的助手。\n
        你的任务是根据下面提供的上下文回答用户的问题。\n
        不要使用任何先前知识或外部信息。\n
        ---------------------\n
        上下文：{context_str}\n
        ---------------------\n
        查询：{query_str}\n
        说明：\n
        1. 仔细阅读并理解提供的上下文。\n
        2. 如果上下文包含足够的信息来回答查询，请提供一个清晰简洁的答案。\n
        3. 不要编造或猜测任何信息。\n
        答案：'''.format(context_str=context_str, query_str=query)

    response = deepseek_api(qa_prompt);
    print("deepseek-r1 回复:", response)

question = "贾宝玉为何出家？"
print(f"问题: {question}")
search(question)