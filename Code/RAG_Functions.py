# RAG_Fuctions

###################################################################################################

# Packages
#import pymilvus
import time

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

    # Get sentences and documents from results
    sentences = []
    companies = []
    documents = []
    for hits in results:
        # Get ids
        #print(hits.ids)
        # Get distances
        #print(hits.distances)
        for hit in hits:
            # Get id
            #print(hit.id)
            # Get distance
            #print(hit.distance) # hit.score
            # Get vector
            #hit.vector
            # Get output field
            #print(hit.get("sentence"))
            sentences.append(hit.get("sentence"))
            companies.append(hit.get("company_name"))
            documents.append(hit.get("document_name"))

    # get the IDs of all returned hits
    #results[0].ids

    # get the distances to the query vector from all returned hits
    #results[0].distances
    #print(results)

    # get the value of an output field specified in the search request.
    #hit = results[0][0]
    #hit.entity.get('sentence')

    # Get filenames
    # Join company and document names on underscore and add .txt
    filenames = [f'{company}_{document}.txt' for company, document in zip(companies, documents)]
    # Keep unique values
    filenames = list(set(filenames))

    # get unique document names
    # documents = list(set(documents))

    # return sentences, filenames, and time taken rouded to 2 decimal places
    return sentences, filenames, end_time - start_time
