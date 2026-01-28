Faiss DB implementation
## Improved Faiss Description

**Faiss** is a high-performance library for **efficient similarity search and clustering** of dense vectors (embeddings). It supports massive datasets—even those exceeding RAM—through advanced indexing algorithms that enable **sub-second searches** across billions of vectors. [stackoverflow](https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists)

## Core Functionality

**Faiss builds specialized indexes** in memory for a set of D-dimensional vectors. Once constructed, it performs **lightning-fast similarity searches** using:

| Distance Metric | Use Case | Formula |
|-----------------|----------|---------|
| **L2 (Euclidean)** | General vector similarity | `||x - y||₂`  [geeksforgeeks](https://www.geeksforgeeks.org/python/how-to-calculate-cosine-similarity-in-python/) |
| **Cosine** | Normalized text embeddings | `1 - cos(x, y)`  [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) |
| **Inner Product** | Recommendation systems | `x · y` |

## Euclidean Distance Explained

**Euclidean distance** measures straight-line distance in vector space:
```
2D: Distance between (1,2) and (4,6) = √((4-1)² + (6-2)²) = 5
High-D: Same principle scales to 768D (BERT) or 1536D embeddings
```

**Smaller distance = more similar.** For word embeddings:
- "pet" ↔ "dog": ~0.3 (close)
- "pet" ↔ "car": ~1.8 (distant)

## Normalization Matters

**Raw embeddings need normalization** for meaningful comparisons:

```python
# Before: Vectors have different magnitudes
embedding1 = [0.1, 0.2, 0.7]  # Magnitude: 0.77
embedding2 = [0.5, 0.3, 0.9]  # Magnitude: 1.22

# L2 Normalization: Scale to unit length
norm1 = embedding1 / np.linalg.norm(embedding1)  # All vectors magnitude = 1.0
# Now cosine similarity = Euclidean distance on normalized vectors
```

## Quick Start Code

```bash
# CPU
pip install faiss-cpu sentence-transformers

# GPU (CUDA 11.8+)
pip install faiss-gpu sentence-transformers
```

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = ["Transformer excels at translation", "BERT loves NLP tasks"]
embeddings = model.encode(texts)  # Shape: (2, 384)

# 2. Build Faiss index
d = embeddings.shape [stackoverflow](https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists)  # Dimension: 384
index = faiss.IndexFlatL2(d)  # L2 distance index
index.add(embeddings.astype('float32'))  # Add vectors

# 3. Search
D, I = index.search(embeddings.astype('float32'), k=2)  # Distance, Indices
print(f"Top matches: {I}")  # [[0,1], [1,0]]
print(f"Distances:  {D}")   # [[0.0, 0.45], [0.45, 0.0]]
```

## Why Faiss?
- **10M vectors**: <1ms/query on CPU
- **1B+ vectors**: IVF/PQ indexes scale linearly
- **Production-ready**: Used by Facebook AI, OpenAI [tigerdata](https://www.tigerdata.com/learn/implementing-cosine-similarity-in-python)

**Perfect for RAG, recommendation systems, and semantic search at scale.**

Figure 4.3: Faiss installation
<img width="919" height="550" alt="image" src="https://github.com/user-attachments/assets/6f66e9e2-577c-42b7-b11b-c062f820f65d" />


Install the sentence:

Figure 4.4: Transformers installation
Larger View
<img width="1141" height="385" alt="image" src="https://github.com/user-attachments/assets/1e63b6ac-4f86-4760-87d7-9698860312d1" />


We will load the same reviews dataset that we used in the previous chapter, as shown in Figure 4.5:

Figure 4.5: Load data
<img width="468" height="301" alt="image" src="https://github.com/user-attachments/assets/a77c6e56-fa4f-4979-8604-13cdf07ef8b4" />

<img width="468" height="301" alt="image" src="https://github.com/user-attachments/assets/e66afb46-457b-47af-9999-2c0c53f14d21" />

Now, we need to convert the text into vectors, as we discussed in the previous chapter. We will use sentence transformers here for the same, as shown in Figure 4.6:

Figure 4.6: Convert to vector
<img width="914" height="613" alt="image" src="https://github.com/user-attachments/assets/116440c8-29b9-471f-b2b5-111fc981a14e" />


The embedding model we used has 768 dimensions, so each vector has a length of 768, as shown in Figure 4.7. In this case, the Euclidean space will also be 768 to calculate the similarity search.

Figure 4.7: Dimension of vectors

The next step is to build a Faiss index from vectors. Since we have 768 dimensions, an L2 distance index is created in 768-dimensional Euclidean space, and L2 normalized vectors are added to that index. In FAISS, an index is an object that helps in efficient similarity search. A Faiss index can be created, as shown in Figure 4.8, to calculate the Euclidean distance.

IndexFlatL2 measures the L2 (or Euclidean) distance between all given points between our query vector and the vectors loaded into the index.


Now, we can search with our query. For example, we can ask how good is the quality of the dress?. As in step 2, a search text is transformed using vectorization. Then, the vector is also normalized because all the vectors within the search index are normalized, as shown in Figure 4.9:

Figure 4.9: Search vector
<img width="851" height="200" alt="image" src="https://github.com/user-attachments/assets/4b3ad402-d3c8-4ef9-b6a5-44b7acd6c809" />


Since we only have five rows in our sample data set, we can set k to the total number of vectors within the index, which is 5. For massive datasets, we can specify the value for k based on which top 5 or 10 similar vectors will be retrieved based on the k value, as shown in Figure 4.10:

Figure 4.10: Setting k value
<img width="471" height="131" alt="image" src="https://github.com/user-attachments/assets/c9231ae8-7ab6-431b-ae0b-52c3bc50a45a" />

Next, we must sort the search results in ascending order. In the result data frame shown in Figure 4.11, the distances are sorted in an ascending order. The src column is the ANN corresponding to that distance, meaning that src 0 is the vector at position 0 in the index. Similarly, src 4 is the vector at position 4 in the index based on the order of text vectors from step 1.

Figure 4.11: Sort results
<img width="585" height="257" alt="image" src="https://github.com/user-attachments/assets/9c8c6942-981d-4f09-b76f-36da82f0fcd1" />


We can merge the search results with our original data frame. For our query, How good is the quality of the dress? it has identified this review: This dress is perfection! so pretty and flattering as the more similar one as shown in Figure 4.12:

Figure 4.12: Search result
<img width="450" height="261" alt="image" src="https://github.com/user-attachments/assets/968a5cc6-b9e8-49af-8ba6-571c808e8b59" />

