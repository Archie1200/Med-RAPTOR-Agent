# Med-RAPTOR Agent
A Generative and Agentic AI system for Evidence-Based medical question answering using RAPTOR-style hierarchical retrieval and NLI-driven verify-and-revise loops.

---

## 📊 Results Summary

| System | F1 Score |
| :--- | :---: |
| Baseline RAG | 20.00% |
| RAPTOR (without NLI) | 60.00% |
| RAPTOR Agent (with NLI) | 80.00% |

### Step 1: Create Conda Environment

conda create -n medraptor python=3.10 -y

conda activate medraptor

### Step 2: Install Dependencies

pip install -r requirements.txt

### Step 3: Run the Code

#### Option 1: Baseline RAG (Vanilla)
python baseline_rag.py

#### Option 2: RAPTOR Agent (without NLI)

python Raptor_NLI.py

#### Option 3: RAPTOR Agent (with NLI) - **Best Performance**
python Raptor_NonNLI.py

## 📈 Outputs
Each script generates:
- **Terminal output**: Round-by-round performance metrics
- **Graph**: `med_raptor_performance.png` (visualization showing accuracy and F1 improvement across agentic rounds)
- **Statistics**: Exact Match (EM) and F1 scores with delta improvements

## Configuration

Default settings:
- **Dataset**: PubMedQA (biomedical yes/no/maybe questions)
- **Index size**: 150 documents
- **Test size**: 30 questions 
- **Agentic rounds**: 5 iterations
- **Models**: 
  - Generation: Google Gemma 2B-IT
  - NLI Verification: DeBERTa-v3-base
  - Embeddings: MPNet sentence transformers


## 📚 Methodology

### 1. Baseline RAG
- Standard retrieval-augmented generation
- ChromaDB vector search → Top-k documents → LLM generation

### 2. RAPTOR (without NLI)
- Hierarchical document indexing with clustering
- Multi-level retrieval (leaves + clusters + summaries)
- Single-shot generation with better context

### 3. RAPTOR Agent (with NLI) 
- **Hierarchical retrieval** from RAPTOR tree
- **Agentic refinement loop** (5 rounds):
  1. Generate answer
  2. Extract factual claims
  3. Verify each claim with NLI (DeBERTa-v3)
  4. Generate feedback for unsupported claims
  5. Revise answer or keep best
- **Cumulative best tracking**: Monotonically improving performance


## For More Details

See the attached report: **`24826_abstract.pdf`**


## Acknowledgments
- RAPTOR framework (Sarthi et al., 2024)
- PubMedQA dataset
- Hugging Face Transformers
- LangChain for LLM orchestration
