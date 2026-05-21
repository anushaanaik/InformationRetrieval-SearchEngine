import math
from collections import defaultdict, Counter
import ir_datasets
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# --- 1. INITIALIZATION & DATA LOADING ---
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

print("Loading Cranfield Dataset...")
dataset = ir_datasets.load("cranfield")
documents = list(dataset.docs_iter())
N = len(documents)

# --- 2. BUILD INDEX & VECTORS (Chapters 1, 2, 5, 6) ---
print("Building Index and Pre-calculating Vectors...")
inverted_index = defaultdict(list)
doc_vectors = defaultdict(dict)
doc_lengths = defaultdict(float)

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return [stemmer.stem(w) for w in tokens if w.isalnum() and w not in stop_words]

# Build basic Boolean Index
for doc in documents:
    doc_id = int(doc.doc_id)
    words = preprocess_text(doc.title + " " + doc.text)
    for word in set(words):
        inverted_index[word].append(doc_id)

# Calculate IDF
idf_dict = {term: math.log10(N / len(postings)) for term, postings in inverted_index.items()}

# Calculate TF-IDF Document Vectors
for doc in documents:
    doc_id = int(doc.doc_id)
    term_counts = Counter(preprocess_text(doc.title + " " + doc.text))
    sum_squares = 0.0
    for term, count in term_counts.items():
        tfidf = (1 + math.log10(count)) * idf_dict[term]
        doc_vectors[doc_id][term] = tfidf
        sum_squares += (tfidf ** 2)
    doc_lengths[doc_id] = math.sqrt(sum_squares)

print("System Ready!")

# --- 3. LEVENSHTEIN SPELLING CORRECTION---
def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]

def autocorrect_query(query_text):
    raw_tokens = word_tokenize(query_text.lower())
    corrected = []
    made_changes = False
    
    for token in raw_tokens:
        if not token.isalnum() or token in stop_words: continue
        stemmed = stemmer.stem(token)
        
        if stemmed not in inverted_index:
            best_match, lowest_dist, highest_freq = stemmed, 2, 0
            for dict_word in inverted_index.keys():
                if abs(len(stemmed) - len(dict_word)) > 2: continue
                dist = levenshtein_distance(stemmed, dict_word)
                if dist <= 2:
                    freq = len(inverted_index[dict_word])
                    if dist < lowest_dist or (dist == lowest_dist and freq > highest_freq):
                        lowest_dist, best_match, highest_freq = dist, dict_word, freq
            if best_match != stemmed:
                corrected.append(best_match)
                made_changes = True
                continue
        corrected.append(stemmed)
    return " ".join(corrected), made_changes

# --- 4. SEARCH ENGINE & ROCCHIO---
def search_engine(query_text, apply_rocchio=False, top_k=5):
    final_query, was_corrected = autocorrect_query(query_text)
    
    query_words = final_query.split()
    query_counts = {}
    for w in query_words:
        query_counts[w] = query_counts.get(w, 0) + 1
        
    query_vector = {}
    for t, c in query_counts.items():
        if t in idf_dict:
            query_vector[t] = float((1 + math.log10(c)) * idf_dict[t])
            
    if not query_vector: 
        return {"error": "No valid terms in query.", "results": [], "total_matches": 0}
    
    def do_search(q_vec):
        q_len = math.sqrt(sum(w**2 for w in q_vec.values()))
        candidates = set()
        for t in q_vec.keys():
            candidates.update(inverted_index[t])
            
        scores_dict = {}
        for doc_id in candidates:
            dot = sum(q_vec[t] * doc_vectors[doc_id].get(t, 0.0) for t in q_vec)
            if doc_lengths[doc_id] > 0 and q_len > 0:
                scores_dict[str(doc_id)] = float(dot / (doc_lengths[doc_id] * q_len))
                
        sorted_docs = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
        
        # FIX: Capture the TOTAL number of matches before slicing
        total_hits = len(sorted_docs)
        
        result_list = []
        # Only take the top_k for display
        for doc_id_str, score in sorted_docs[:top_k]:
            result_list.append({"id": int(doc_id_str), "score": score})
            
        return result_list, total_hits

    base_results, total_matches = do_search(query_vector)
    
    rocchio_terms = []
    if apply_rocchio and base_results:
        rel_docs = [res["id"] for res in base_results]
        centroid = {}
        
        for d in rel_docs:
            for t, w in doc_vectors[d].items(): 
                centroid[t] = centroid.get(t, 0.0) + float(w)
                
        for t in centroid: 
            centroid[t] = float((centroid[t] / len(rel_docs)) * 0.75)
        
        combined_vector = {}
        for t in query_vector:
            combined_vector[t] = float(query_vector[t])
        for t in centroid:
            combined_vector[t] = float(combined_vector.get(t, 0.0) + centroid[t])
            
        original_terms = set(final_query.split())
        candidate_new_terms = {}
        for t, w in combined_vector.items():
            if t not in original_terms:
                candidate_new_terms[str(t)] = float(w)
                
        sorted_new_terms = sorted(candidate_new_terms.items(), key=lambda x: x[1], reverse=True)[:2]
        rocchio_terms = [str(t[0]) for t in sorted_new_terms]
        
        # Search again with expanded query
        base_results, total_matches = do_search(combined_vector)

    formatted_results = []
    for res in base_results:
        doc_id = res["id"]
        score = res["score"]
        doc_title, doc_snippet = "Unknown", ""
        
        for d in documents:
            if int(d.doc_id) == doc_id:
                doc_title = str(d.title).strip()
                doc_snippet = str(d.text)[:200] + "..."
                break
                
        formatted_results.append({
            "id": int(doc_id), 
            "score": float(round(score, 4)), 
            "title": doc_title, 
            "snippet": doc_snippet
        })
        
    return {
        "corrected_query": str(final_query) if was_corrected else None,
        "rocchio_terms": rocchio_terms,
        "results": formatted_results,
        "total_matches": total_matches # Exporting the true count
    }

# --- 5. FASTAPI WEB SERVER ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})

@app.post("/", response_class=HTMLResponse)
async def handle_search(request: Request, query: str = Form(...), use_rocchio: bool = Form(False)):
    try:
        search_data = search_engine(str(query), apply_rocchio=use_rocchio, top_k=10)
        
        safe_corrected_query = str(search_data["corrected_query"]) if search_data.get("corrected_query") else None
        safe_rocchio_terms = [str(term) for term in search_data.get("rocchio_terms", [])]
        safe_error = str(search_data["error"]) if search_data.get("error") else None
        safe_total = int(search_data.get("total_matches", 0))
        
        safe_results = []
        if search_data.get("results"):
            for res in search_data["results"]:
                safe_results.append({
                    "id": int(res["id"]), "score": float(res["score"]),
                    "title": str(res["title"]), "snippet": str(res["snippet"])
                })
                
        return templates.TemplateResponse(
            request=request,
            name="index.html", 
            context={
                "original_query": str(query),
                "rocchio_checked": bool(use_rocchio),
                "corrected_query": safe_corrected_query,
                "rocchio_terms": safe_rocchio_terms,
                "error_msg": safe_error,
                "results": safe_results,
                "total_matches": safe_total # Passed to HTML
            }
        )
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return templates.TemplateResponse(
            request=request, name="index.html", 
            context={"original_query": str(query), "error_msg": f"Server Error: {str(e)}", "results": []}
        )