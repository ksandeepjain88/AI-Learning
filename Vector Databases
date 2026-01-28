# Types of Vector Databases

There are different types of vector databases, as shown in **Figure 4.2**, and they can be grouped into the following broad categories:

- Vector libraries like Faiss, Annoy, and HNSWLib
- Full-text search databases like Elasticsearch
- Vector-only databases like Pinecone, Chroma
- Vector capable NoSQL databases like MongoDB and Cassandra
- Vector-capable SQL databases like SingleStoreDB

**Figure 4.2:** *Vector DB types source (SingleStore)*

## Popular Vector Databases and Their Key Features

- **Chroma:** A database for building AI applications with embeddings. It employs a unique approach that combines log-structured storage with incremental indexing to achieve efficient retrieval and scalability.
- **Milvus:** An open-source vector database optimized for large-scale vector similarity search and clustering. It is built to power embedding similarity search and AI applications, making unstructured data search more accessible.
- **Pinecone:** A cloud-native vector database offering scalability and flexibility for real-time vector search applications. It provides long-term memory for high-performance AI applications with low latency at scale.
- **Weaviate:** A semantic vector search engine combining vector search with knowledge graph capabilities. It stores objects and vectors, allowing structured filtering via GraphQL, REST, and various language clients.
- **SingleStore:** A cloud-native distributed relational SQL database supporting efficient vector storage and retrieval.
- **Elasticsearch:** A distributed RESTful search and analytics engine capable of addressing various use cases with fast search, relevancy tuning, and scalable analytics.
- **Amazon Neptune:** Fully managed graph database supporting vector embeddings and graph-based algorithms.
- **Neo4j:** Cloud-based graph database with native support for vector embeddings; stores nodes, edges, attributes.
- **OrientDB Graph Edition:** Open-source graph database providing document-oriented and graph-based storage for vectors.
- **YugaByte DB:** Distributed SQL database supporting vector data types and graph extensions for analysis.

## Comparison of Two Types of Vector Databases
| | Indexed Vector DBs | Graph Vector DBs |
|---|---|---|
| Description | Store vector embeddings in a structured format using indexing techniques such as KNN or hierarchical structures to accelerate searches | Represent vectors as nodes in a graph; utilize graph algorithms for relationships |
| Use Cases | Efficient similarity searches: recommendation systems, anomaly detection, fraud detection, content search, knowledge graphs, social network analysis, semantic search | Understanding relationships: recommendation systems based on user preferences |
| Examples | ChromaDB, Milvus, Pinecone, Weaviate | Amazon Neptune, Neo4j |

## Factors to Consider When Choosing a Vector Database
1. Data types: What vectors are you storing (text, images)?
2. Scale: How much data do you need to store/query?
3. Performance: How quickly do you need to perform searches?
4. Features: Do you need advanced features like anomaly detection or semantic search?
5. Deployment: Cloud-based or standalone?

### Recommendations Based on Application Needs:
- For efficient search/retrieval from knowledge bases — use indexed vector databases like Pinecone, Chroma, Weaviate.
- For understanding relationships/connections — opt for graph vector databases like Neptune or Neo4j.
