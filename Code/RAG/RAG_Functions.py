# RAG_Fuctions

###################################################################################################

# Packages
import time

###################################################################################################

def get_mixedbread_of_query(model, query: str):
    '''
    Returns mixedbread embedding for an input text. Text is appropriately formatted to be a query.

    Parameters:
    - model: embedding model
    - query: str: The query to be transformed.
    '''
    transformed_query = f'Represent this sentence for searching relevant passages: {query}'
    return model.encode(transformed_query)

def return_top_5_sentences(collection, query_embedding):
    '''
    Returns top 5 sentences from the collection based on the query embedding. Also includes unique associated files used and time taken.
    '''

    # Set search parameters
    search_params = {
        "metric_type": "L2", # simliarity metric
        "offset": 0, # the number of top-k hits to skip
        "ignore_growing": False#, # ignore growing segments
        #"params": {"nprobe": 10} # number of cluster units, only used for some index types
    }

    # Start timer
    start_time = time.time()

    # Use Milvus to search for similar vectors
    results = collection.search(
        data=[query_embedding], # query vector
        anns_field="embedding", # name of field to search on
        param=search_params, # seach parameters set above
        limit=5,# num results to return
        expr=None, # boolean filter
        output_fields=['company_name', 'sentence', 'document_name'], # fields to return 
        consistency_level="Strong"
    )

    # End timer
    end_time = time.time()

    # Get sentences, companies, and documents from results
    sentences = []
    companies = []
    documents = []
    for hits in results:
        for hit in hits:
            sentences.append(hit.get("sentence"))
            companies.append(hit.get("company_name"))
            documents.append(hit.get("document_name"))

    # Get filenames
    # Join company and document names on underscore and add .txt
    filenames = [f'{company}_{document}.txt' for company, document in zip(companies, documents)]
    # Keep unique values
    filenames = list(set(filenames))

    # Return sentences, filenames, and time taken
    return sentences, filenames, end_time - start_time
