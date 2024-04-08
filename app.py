from pymilvus import connections, Collection

connections.connect(host='localhost', port='19530')
collection_name = 'movies'

collection = Collection(name=collection_name)
query_expression = "description_id in [448952306084872222]"
query_results = collection.query(expr=query_expression, output_fields=["description_id", "vector"])

for result in query_results:
  print(f"ID: {result['description_id']}, Vector: {result['vector']}")
