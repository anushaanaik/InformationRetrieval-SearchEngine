# Information Retrieval System & Search Engine

A fully functional, from-scratch Information Retrieval (IR) system built using Python and FastAPI. This project implements the core mechanics of modern search engines without relying on external search frameworks (like Elasticsearch or Lucene). 

It was developed and evaluated using the classic **Cranfield aerodynamics dataset** (1,400 documents).

## Core Features Implemented

* **Boolean Indexing:** Tokenization, stopword removal, and Porter Stemming via NLTK to build an efficient Inverted Index.
* **Tolerant Retrieval:** Implementation of the **Levenshtein Edit Distance** algorithm to catch user typos and provide "Did you mean...?" spelling corrections.
* **Index Compression:** Significant memory reduction via Delta Encoding and **Variable-Byte (V-Byte) Compression** applied to document postings lists.
* **Vector Space Ranking:** Documents are ranked based on their **TF-IDF** weights and **Cosine Similarity** to the query vector.
* **Index Elimination:** Query processing is optimized by calculating dot products exclusively for documents containing query terms.
* **Query Expansion:** Integration of the **Rocchio Algorithm** for Pseudo-Relevance Feedback. The system calculates the vector centroid of top-retrieved documents to automatically inject missing, highly-relevant terminology into the user's query.

## 🚀 Tech Stack
* **Backend Core:** Python 3
* **NLP Processing:** NLTK (Natural Language Toolkit)
* **API & Server:** FastAPI, Uvicorn
* **Frontend UI:** Jinja2 HTML Templates, CSS

## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/information-retrieval-search-engine.git
   cd information-retrieval-search-engine
   ```

2. **Install the required dependencies:**
   ```bash
   pip install fastapi uvicorn ir_datasets nltk jinja2
   ```

3. **Start the FastAPI Server:**
   ```bash
   uvicorn app:app --reload
   ```

4. **Access the Web Interface:**
   Open your browser and navigate to: `http://127.0.0.1:8000`

*Note: On the first run, the server will automatically download the 1,400-document Cranfield dataset and build the initial Inverted Index and TF-IDF vectors in memory.*

## 📊 Evaluation & Metrics
The system was evaluated against the Cranfield collection's 225 ground-truth relevance queries. By balancing the Precision-Recall trade-off, the engine establishes a strong baseline **Mean Average Precision (MAP)** using standard Cosine Similarity, which is visibly and mathematically improved when the Rocchio Query Expansion feature is enabled.