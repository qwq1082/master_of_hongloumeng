# import
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core import StorageContext

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# define embedding function
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)


# load documents
documents = SimpleDirectoryReader(r"E:\code\rag\data").load_data()
print(documents[:5])
print(len(documents[0].text_resource.text))

# define index
vector_store = ElasticsearchStore(
    es_url="http://localhost:9200",  # see Elasticsearch Vector Store for more authentication options
    index_name="hlm",
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

'''
# Query Data
query_engine = index.as_query_engine()
response = query_engine.query("总结一下黛玉的剧情")
print(response)
'''