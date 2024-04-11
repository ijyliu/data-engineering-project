# RAG_Fuctions

###################################################################################################

# Packages
#import pymilvus

###################################################################################################

def transform_query(query: str) -> str:
    """ 
    For mixedbread embedding retrieval, add the prompt for query (not for documents).
    """
    return f'Represent this sentence for searching relevant passages: {query}'

def get_mixedbread_of_query(model, input: str):
    '''
    Returns mixedbread embedding for an input text. Text is appropriately formatted to be a query.
    '''
    transformed_input = transform_query(input)
    return model.encode(transformed_input)

def return_top_5_sentences(client, query_embedding):
    res = client.search(
        collection_name="text_embeddings",
        data=[query_embedding],
        limit=5, # Max. number of search results to return
        search_params={"metric_type": "COSINE", "params": {}}, # Search parameters
        output_fields = ["sentence", "company_name", "document_name"] # Output fields
    )
    return res
