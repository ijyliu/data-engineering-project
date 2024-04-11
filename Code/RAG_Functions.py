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

def return_top_5_sentences(collection, query_embedding):
    search_params = {
        "metric_type": "L2", # simliarity metric
        "offset": 0, # the number of top-k hits to skip
        "ignore_growing": False#, # ignore growing segments
        #"params": {"nprobe": 10} # number of cluster units, only used for some index types
    }
    results = collection.search(
        data=[query_embedding], # query vector
        anns_field="embedding", # name of field to search on
        param=search_params, # seach parameters set above
        limit=5,# num results to return
        expr=None, # boolean filter
        output_fields=['company_name', 'sentence'], # fields to return 
        consistency_level="Strong"
    )

    # Get sentences from results
    sentences = []
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

    # get the IDs of all returned hits
    #results[0].ids

    # get the distances to the query vector from all returned hits
    #results[0].distances
    #print(results)

    # get the value of an output field specified in the search request.
    #hit = results[0][0]
    #hit.entity.get('sentence')
    return sentences
