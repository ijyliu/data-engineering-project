#simple query testing, milvus is trash at this tho its good at vector similarity lookup
from pymilvus import connections, utility, Collection

connections.connect(host='localhost', port='19530')

collection_name = 'text_embeddings'
collection = Collection(name=collection_name)
# literally no idea what this does but you need it
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}
collection.create_index(field_name="embedding", index_params=index_params)
collection.load()

expr = f"company_name == 'TheHersheyCompany'"

results = collection.query(expr=expr)

for result in results:
    print(result)

    