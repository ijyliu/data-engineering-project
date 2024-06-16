# Data Engineering Project

Fangyuan Li, Isaac Liu, Robert Thompson 

In this paper, we set up and benchmarked the popular open-source vector database system Milvus for sentence queries from the TOSDR (Terms of Service; Didn't Read) corpus. After creating vector collections running in Docker-Compose configurations, we loaded the text of 1,623 documents (several GB) represented as 1,024-dimensional Mixedbread.ai retrieval embeddings. We then benchmarked the performance of various search types (range search, bulk-vector search, etc.) and similarity indices (exhaustive, clustered nearest neighbors, hierarchical, quantized), examining tradeoffs in returned result distance, index creation, and query time.

We also set the foundations for using our embeddings and Milvus for a full Retrieval Augmented Generation system, where a user's query could be matched with relevant information for a language model conversation. We first experimented with a local-running FLAN-T5 model as the chatbot LM, but ultimately opted to use Google Gemini to construct a prototype. Isaac continued his work on this to develop a web app [here](https://github.com/ijyliu/milvus-rag-web-app).

Our report can be found [here](https://docs.google.com/document/d/1MmpETAMZjmVuSV9vUHBsQRk6wCEzYUA8m-2mjfX16VY/edit).

## Technologies (not exhaustive!)

- Milvus
- Docker-Compose
- Python
  - PyMilvus SDK
  - Transformers
  - Sentence-Transformers on GPU
  - NLTK
  - Pandas
  - Google Generative AI API
  - Conda
