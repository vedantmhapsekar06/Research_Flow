# ResearchFlow

A fully local, hybrid-retrieval RAG chatbot for querying multiple research papers — no paid APIs, no cloud dependency. Upload PDFs, ask natural-language questions, and get answers grounded in the actual paper text with page-level citations.

Built as a portfolio project to explore retrieval-augmented generation end-to-end: chunking strategy, hybrid semantic + keyword search, citation grounding, and hallucination resistance — all running on a local LLM via Ollama.

---

## Why Local?

Every part of this stack runs on your own machine, for free:

- **LLM + Embeddings:** Ollama (`llama3.2` for chat, `nomic-embed-text` for embeddings)
- **Vector Store:** ChromaDB (persisted to disk)
- **Keyword Search:** BM25 (`rank_bm25`)
- **PDF Parsing:** PyMuPDF
- **Backend:** FastAPI
- **Frontend:** HTML, CSS, JavaScript

No API keys, no per-request cost, and no research documents need to leave your machine.

---

## Architecture

```text
                         RESEARCHFLOW
                              │
                              ▼
                       PDF Upload
                              │
                              ▼
                  Page-Level Text Extraction
                              │
                              ▼
                Overlapping Chunking
                 220 words / 40 overlap
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
             Ollama Embeddings       BM25
            nomic-embed-text      Keyword Index
                    │                   │
                    ▼                   ▼
                ChromaDB          Keyword Search
                    │                   │
                    └─────────┬─────────┘
                              ▼
                       Hybrid Retrieval
                      Semantic + BM25
                         Score Fusion
                           0.6 / 0.4
                              │
                              ▼
                     Top-K Relevant Chunks
                              │
                              ▼
                         Llama 3.2
                              │
                              ▼
                  Grounded Answer + Sources
                              │
                              ▼
                    ResearchFlow Chat UI
````

### Query Pipeline

```text
User Question
      │
      ▼
Conversation Context
      │
      ▼
Follow-Up Query Resolution
      │
      ▼
Standalone Search Query
      │
      ▼
Hybrid Retrieval
 ┌────┴────┐
 ▼         ▼
Semantic   BM25
Search     Search
 └────┬────┘
      ▼
Weighted Score Fusion
      │
      ▼
Top-K Chunks
      │
      ▼
Llama 3.2
      │
      ▼
Grounded Response
      │
      ▼
Page-Level Citations
```

**Backend:** FastAPI, serving the REST API and static frontend from a single process.

**Frontend:** Single-file HTML/CSS/JavaScript with no framework and no build step.

---

## Features

### Multi-Paper RAG

* Upload multiple research papers.
* Process papers asynchronously.
* Extract text page-by-page.
* Split documents into overlapping chunks.
* Generate embeddings locally.
* Store embeddings in ChromaDB.
* Build a BM25 keyword index.

### Hybrid Retrieval

ResearchFlow combines two retrieval strategies:

**Semantic Retrieval**

Uses vector embeddings to find conceptually similar content, even when the wording differs.

**BM25 Keyword Retrieval**

Captures exact terminology, acronyms, numbers, and technical keywords.

The two retrieval results are combined using weighted score fusion:

```text
Final Score = 0.6 × Semantic Score
            + 0.4 × BM25 Score
```

This allows ResearchFlow to benefit from both semantic similarity and exact keyword matching.

### Conversational Memory

ResearchFlow supports follow-up questions.

For example:

```text
User:
What retrieval method does Paper A use?

User:
Why did they choose that approach?

User:
How is it different from Paper B?
```

The system uses conversation context to resolve references and converts follow-up questions into standalone retrieval queries before searching the knowledge base.

### Page-Level Citations

Answers include the paper and page associated with the retrieved evidence.

This allows users to verify where an answer came from instead of relying on unsupported model-generated information.

### Hallucination Resistance

The LLM is explicitly instructed to answer only from retrieved paper evidence.

If the requested information cannot be found in the uploaded papers, the system should respond that the information was not found rather than relying on the model's general knowledge.

---

## Technology Stack

| Component            | Technology              |
| -------------------- | ----------------------- |
| Programming Language | Python                  |
| Backend              | FastAPI                 |
| Frontend             | HTML / CSS / JavaScript |
| LLM                  | Llama 3.2               |
| Local Model Runtime  | Ollama                  |
| Embedding Model      | nomic-embed-text        |
| Vector Database      | ChromaDB                |
| Keyword Retrieval    | BM25                    |
| PDF Processing       | PyMuPDF                 |
| API Server           | Uvicorn                 |

---

## Setup

### Prerequisites

* Python 3.10+
* Ollama
* Windows / Linux / macOS
* Recommended: at least 8 GB RAM for comfortable local operation

Install Ollama from:

[https://ollama.com](https://ollama.com)

---

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ResearchFlow.git
cd ResearchFlow
```

---

### 2. Install Ollama Models

Pull the chat model:

```bash
ollama pull llama3.2
```

Pull the embedding model:

```bash
ollama pull nomic-embed-text
```

Make sure Ollama is running before starting ResearchFlow.

---

### 3. Create a Virtual Environment

Navigate to the backend:

```bash
cd backend
```

Create the environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

---

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 5. Start ResearchFlow

```bash
python main.py
```

The application runs on:

```text
http://localhost:8500/
```

Open the URL in your browser.

---

## Usage

### Step 1 — Upload Papers

Upload one or more research papers in PDF format.

```text
Paper 1.pdf
Paper 2.pdf
Paper 3.pdf
...
```

ResearchFlow processes each paper independently.

---

### Step 2 — Processing

Each uploaded paper goes through:

```text
PDF
 ↓
Page Extraction
 ↓
Text Cleaning
 ↓
Chunking
 ↓
Embedding Generation
 ↓
ChromaDB
 ↓
BM25 Index
```

Each chunk retains metadata such as:

* Paper ID
* Paper name
* Page number
* Chunk ID
* Text content

---

### Step 3 — Ask Questions

Ask questions naturally about the uploaded research papers.

Examples:

```text
What is the main problem addressed by this paper?
```

```text
What retrieval method does the paper propose?
```

```text
What chunk size does the system use?
```

```text
What were the reported results?
```

---

### Step 4 — Ask Follow-Up Questions

ResearchFlow maintains conversational context.

Example:

```text
User:
What retrieval approach does Paper A use?

Assistant:
Paper A uses hybrid retrieval...

User:
Why did they use it?

Assistant:
They use it to combine...

User:
How is that different from Paper B?

Assistant:
Paper B differs by...
```

The follow-up questions are reformulated into standalone search queries before retrieval.

---

## Multi-Paper Questions

ResearchFlow can retrieve evidence from multiple papers for a single question.

Example:

```text
Compare the retrieval approaches used in these papers.
```

The system may retrieve:

```text
Paper A → Chunk 12
Paper A → Chunk 18
Paper B → Chunk 7
Paper C → Chunk 21
```

The retrieved evidence is then provided to the LLM to generate a combined answer.

---

## Testing Methodology

Rather than relying only on manual testing, ResearchFlow was tested against several RAG-specific failure modes.

The test suite focuses on six categories:

| # | Category                 | What It Verifies                                  | Result       |
| - | ------------------------ | ------------------------------------------------- | ------------ |
| 1 | Basic Retrieval          | Simple single-paper factual questions             | ✅ Pass       |
| 2 | Exact Factual Retrieval  | Precise numbers survive chunking and retrieval    | ✅ Pass       |
| 3 | Cross-Paper Retrieval    | Information can be synthesized from multiple PDFs | ✅ Pass       |
| 4 | Follow-Up Questions      | Conversational context resolves references        | ✅ Pass       |
| 5 | Citation Accuracy        | Cited page contains the claimed information       | ✅ Pass       |
| 6 | Hallucination Resistance | System refuses unsupported/fabricated premises    | ✅ Pass       |

---

## Example Test Questions

### Basic Retrieval

```text
What is the main problem addressed by this paper?
```

### Exact Retrieval

```text
What chunk size and overlap does the paper use?
```

### Numerical Retrieval

```text
What citation accuracy was reported for the hybrid retrieval approach?
```

### Cross-Paper Retrieval

```text
Compare the approaches used by these two papers.
```

### Follow-Up Retrieval

```text
What retrieval method does the first paper use?
```

Follow-up:

```text
Why did they use that approach?
```

Follow-up:

```text
How is it different from the second paper?
```

### Hallucination Test

Ask about information that does not exist in the uploaded papers:

```text
Does the paper use Redis for caching?
```

The expected behavior is:

```text
The information was not found in the uploaded papers.
```

rather than an answer generated from the LLM's general knowledge.

---


## Known Limitations

### 1. Table and Figure Structure

PDF text extraction can flatten tables into linear text.

For example, a table such as:

```text
PaperQA    100%    0%    0%
Vanilla RAG 22%    6%   22%
```

may be extracted as a sequence of values without preserving the original row-column relationships.

This can make dense results tables difficult for a small local LLM to interpret correctly.

This limitation originates primarily from the PDF extraction stage rather than the LLM itself.

Dedicated table extraction or document-understanding models could improve this in future versions.

---

### 2. Retrieval Recall on Broad Questions

Broad comparative questions can be more difficult than narrow factual questions.

For example:

```text
Compare Paper A and Paper B's approaches.
```

Relevant evidence may be distributed across multiple non-adjacent chunks.

As a result, a relevant chunk may not always appear in the final Top-K retrieval results.

This is a retrieval-recall limitation rather than simply an LLM limitation.

Possible improvements include:

* Increasing retrieval Top-K
* Adding a reranking stage
* Using a cross-encoder reranker
* Query expansion
* Multi-query retrieval

---

### 3. Small Local LLM

ResearchFlow currently uses:

```text
llama3.2
```

The model performs well for single-source and multi-source synthesis in testing, but smaller local models can require more explicit prompting and grounding instructions.

A larger local model can be substituted if additional computational resources are available.

For example:

```text
llama3.1:8b
```

The rest of the RAG pipeline can remain largely unchanged.

---

### 4. In-Memory Conversation History

Conversation history is currently maintained in memory.

Therefore:

```text
Server restart
      ↓
Conversation history reset
```

Persistent conversation storage is a potential future improvement.

---

### 5. PDF Viewer

The current source panel displays relevant excerpts and page information.

An integrated PDF viewer with direct page navigation could improve source verification.

---

## Project Structure

```text
ResearchFlow/
│
├── backend/
│   │
│   ├── main.py
│   │
│   ├── core/
│   │   └── config.py
│   │
│   ├── models/
│   │   └── schemas.py
│   │
│   ├── services/
│   │   ├── pdf_processor.py
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   ├── storage.py
│   │   ├── retriever.py
│   │   ├── llm.py
│   │   └── chat_service.py
│   │
│   ├── api/
│   │   ├── papers.py
│   │   └── chat.py
│   │
│   └── requirements.txt
│
├── frontend/
│   └── index.html
│
├── data/
│   └── uploaded_papers/
│
├── vector_store/
│
└── README.md
```

---

## Core Components

### `pdf_processor.py`

Responsible for:

```text
PDF
 ↓
Page-by-page text extraction
 ↓
Clean page text
 ↓
Page metadata
```

---

### `chunker.py`

Splits extracted page text into overlapping chunks.

Current configuration:

```text
Chunk size: ~220 words
Overlap: ~40 words
```

The overlap helps preserve context between neighboring chunks.

---

### `embeddings.py`

Uses Ollama's:

```text
nomic-embed-text
```

to convert chunks into vector representations.

---

### `vector_store.py`

Stores and retrieves embeddings using ChromaDB.

Each stored chunk contains metadata that allows the system to trace retrieved evidence back to the original paper and page.

---

### `retriever.py`

Implements hybrid retrieval:

```text
Semantic Search
      +
BM25 Search
      ↓
Score Fusion
      ↓
Top-K Results
```

Current weighting:

```text
Semantic = 0.6
BM25     = 0.4
```

---

### `llm.py`

Handles communication with the local Ollama LLM.

Responsibilities include:

* Answer generation
* Follow-up query condensation
* Grounding instructions
* Local inference

---

### `chat_service.py`

Acts as the main orchestration layer:

```text
User Question
      ↓
Conversation Context
      ↓
Query Resolution
      ↓
Retrieval
      ↓
Context Construction
      ↓
LLM
      ↓
Answer + Sources
```

---

## Design Philosophy

ResearchFlow follows three main principles.

### 1. Local-First

Research papers remain on the user's machine.

No external API is required for inference or embeddings.

### 2. Retrieval Before Generation

The LLM is not expected to know the answer from its training data.

Relevant evidence is retrieved from the uploaded papers first.

```text
Retrieve
   ↓
Ground
   ↓
Generate
```

### 3. Verifiable Answers

Answers should be traceable back to the source material through:

```text
Paper
+
Page
+
Retrieved excerpt
```

This makes the system more useful for research-oriented question answering.


## License

This project is intended for educational and portfolio purposes.

```
```
