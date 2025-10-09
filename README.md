# 📚 RadLab Article‑Creator  

A modular research framework for large‑scale text analysis: it ingests heterogeneous corpora, 
applies multilingual preprocessing, computes dense embeddings, performs similarity‑based clustering, 
creates LLM‑driven abstractive summaries, and continuously updates a temporal knowledge‑graph 
that captures topic evolution and inter‑document relations.

---

## 1. Project Overview  

`radlab‑article‑creator` is a **Django‑based research platform** that ingests plain‑text items 
(with minimal metadata) and produces a rich set of artifacts:

| Artefact                  | Description                                                                                  | Persistence                                           |
|---------------------------|----------------------------------------------------------------------------------------------|-------------------------------------------------------|
| **Embeddings**            | Vector representation of each news piece (torch‑based embedder)                              | In‑memory, optional on‑disk cache                     |
| **Clusters**              | Groups of semantically similar items (`RdlClusterer`)                                        | Django model `Cluster`                                |
| **GenAI Labels**          | Human‑readable Polish category names generated via an external LLM                           | Stored in `Cluster.genai_label`                       |
| **Summary Articles**      | One‑page Polish summary per cluster (spell‑checked)                                          | Stored in `Cluster.article_text`                      |
| **Day‑to‑Day Similarity** | Cosine similarity links between clusters of adjacent days                                    | Model `SimilarClusters`                               |
| **Hyper‑graph**           | Weighted graph where nodes are day‑wide clusters and edges encode similarity scores          | Serialized `*.pkl` files in the *hyper_graphs* folder |
| **REST API**              | Public and admin endpoints to expose clusters, articles, similarity links, and system status | `creator/api/public/` & `creator/api/admin/`          |

All results are persisted in a **PostgreSQL** (or any Django‑supported) database, making them instantly consumable 
by downstream services such as front‑ends, search engines, or analytics dashboards.

---  

## 2. Key Features  

- **Embedding Layer** – a configurable Torch model turns any text into a fixed‑size vector.  
- **RdlClusterer** – a reduction‑aware, density‑based clustering algorithm that automatically selects a suitable number of clusters within user‑defined bounds.  
- **GenAI Labelling & Summarisation** – local LLM api used to generate concise Polish category names and one‑page summary articles for each cluster.  
- **Temporal Similarity** – cosine similarity between clusters of consecutive days, enabling trend detection and story continuity tracking.  
- **Hyper‑graph Generation** – builds a weighted graph of day‑wide clusters for advanced visualization and network analysis.  
- **Config‑driven Pipeline** – all runtime options (embedder path, clustering parameters, GenAI endpoint, etc.) live in JSON files under `configs/`.  
- **REST API** – clean, versioned endpoints expose clusters, articles, similarity links, and admin utilities.  
- **Extensible Architecture** – swap the embedder, clustering algorithm, or GenAI provider by editing configuration only; code is organised into isolated components for easy testing and replacement.  

---  


## 3. License  

The source code is released under the same license as the repository’s root 
[LICENSE](LICENSE) file (Apache 2.0). See [LICENSE](LICENSE) for the full legal text.
