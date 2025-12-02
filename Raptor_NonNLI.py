"""
Med-RAPTOR Agent (RAPTOR + Gemma 2B, NO NLI LOOP)
Compatible with test.py:
- Same dataset: PubMedQA ("pqa_labeled", train split)
- Same embedding: sentence-transformers/multi-qa-mpnet-base-cos-v1
- Same LLM: google/gemma-2b-it (HuggingFace + LangChain)
- Same RAPTOR library

Differences:
- NO NLI verification
- NO claim extraction/revision loop
- Single-pass QA on RAPTOR retrieval
- UPDATED: Modern LangChain API (LCEL)
"""

import os
import sys
import time
import traceback
import builtins
import re
import string
from collections import Counter
from typing import List, Dict, Any

import torch
from datasets import load_dataset
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# LangChain imports - Modern API
try:
    from langchain.prompts import PromptTemplate
    from langchain.llms import HuggingFacePipeline
    print("[INFO] LangChain successfully imported")
except ImportError as e:
    print("[ERROR] LangChain required: pip install langchain langchain-community langchain-core")
    sys.exit(1)

# RAPTOR imports
try:
    from raptor import (
        BaseSummarizationModel,
        BaseQAModel,
        BaseEmbeddingModel,
        RetrievalAugmentationConfig,
    )
    from raptor.RetrievalAugmentation import RetrievalAugmentation
except Exception as e:
    print("[ERROR] Could not import raptor package.")
    traceback.print_exc()
    sys.exit(1)

original_input = builtins.input
def auto_yes_input(prompt=""):
    print(f"{prompt}y (auto-answered)")
    return "y"
builtins.input = auto_yes_input


os.environ["RPTREE_AUTO_OVERWRITE"] = "1"
os.environ["RAPTOR_SUPPRESS_PROMPTS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

CUDA_AVAILABLE = torch.cuda.is_available()

DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
print(f"[INFO] Device: {DEVICE}")

if CUDA_AVAILABLE:
    print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


def clean_gemma_output(text: str) -> str:
    """Remove prompt artifacts from Gemma output"""
    if not text:
        return text
    
    text = text.split('<start_of_turn>model')[-1] if '<start_of_turn>model' in text else text
    text = text.replace('<end_of_turn>', '')
    text = text.replace('<start_of_turn>user', '')
    
    prefixes = ['Answer:', 'Revised answer:', 'Summary:', 'Context:', 'Question:']
    for prefix in prefixes:
        if text.strip().startswith(prefix):
            text = text.split(prefix, 1)[1]
    
    text = ' '.join(text.split())
    return text.strip()

class ModelManager:
    """Singleton to manage model loading"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        if self.initialized:
            return
        self.gemma_pipeline = None
        self.langchain_llm = None
        self.embedding_model = None
        self.initialized = True
    
    def get_gemma_pipeline(self):
        if self.gemma_pipeline is None:
            print("[INFO] Loading Gemma 2B-IT (full precision)...")
            model_name = "google/gemma-2b-it"
            device_idx = 0 if CUDA_AVAILABLE else -1
            
            self.gemma_pipeline = pipeline(
                "text-generation",
                model=model_name,
                device=device_idx,
                torch_dtype=torch.float16 if CUDA_AVAILABLE else torch.float32,
                max_new_tokens=512
            )
            print("[INFO] Gemma 2B loaded successfully")
        return self.gemma_pipeline
    
    def get_langchain_llm(self):
        if self.langchain_llm is None:
            print("[INFO] Creating LangChain wrapper for Gemma 2B...")
            pipeline_obj = self.get_gemma_pipeline()
            self.langchain_llm = HuggingFacePipeline(pipeline=pipeline_obj)
            print("[INFO] LangChain LLM wrapper created")
        return self.langchain_llm
    
    def get_embedding_model(self):
        if self.embedding_model is None:
            print("[INFO] Loading embedding model (multi-qa-mpnet)...")
            self.embedding_model = SentenceTransformer(
                "sentence-transformers/multi-qa-mpnet-base-cos-v1"
            )
        return self.embedding_model

model_manager = ModelManager()


class GemmaLangChainSummarizationModel(BaseSummarizationModel):
    def __init__(self):
        self.llm = None
        self.prompt = None
    
    def summarize(self, context, max_tokens=150):
        if self.llm is None:
            self.llm = model_manager.get_langchain_llm()
            self.prompt = PromptTemplate.from_template(
                """<start_of_turn>user
Summarize the following medical text concisely, preserving key facts and findings:

{text}
<end_of_turn>
<start_of_turn>model
Summary:"""
            )
        
        try:
            # Use LCEL (LangChain Expression Language)
            chain = self.prompt | self.llm
            summary = chain.invoke({"text": context[:2000]})
            summary = clean_gemma_output(summary)
            return summary.strip()
        except Exception as e:
            print(f"[WARN] Summarizer failed: {e}")
            traceback.print_exc()
            return context[:500]

class GemmaLangChainQAModel(BaseQAModel):
    def __init__(self):
        self.llm = None
        self.prompt = None
    
    def answer_question(self, context, question):
        if self.llm is None:
            self.llm = model_manager.get_langchain_llm()
            self.prompt = PromptTemplate.from_template(
                """<start_of_turn>user
Context: {context}

Question: {question}

Answer concisely based on the context above.
<end_of_turn>
<start_of_turn>model
Answer:"""
            )
        
        try:
            chain = self.prompt | self.llm
            answer = chain.invoke({"context": context[:1500], "question": question})
            answer = clean_gemma_output(answer)
            return answer.strip()
        except Exception as e:
            print(f"[WARN] QA failed: {e}")
            traceback.print_exc()
            return "Unable to generate answer."

class SBertEmbeddingModel(BaseEmbeddingModel):
    def create_embedding(self, text):
        model = model_manager.get_embedding_model()
        return model.encode(text)


def normalize_context(raw_context):
    if isinstance(raw_context, list):
        return " ".join(raw_context)
    if isinstance(raw_context, dict):
        parts = []
        for v in raw_context.values():
            if isinstance(v, list):
                parts.append(" ".join(v))
            else:
                parts.append(str(v))
        return " ".join(parts)
    return str(raw_context)

def build_raptor_index(RA, max_docs=100, force_rebuild=False):
    """Build RAPTOR tree using batch addition"""
    if hasattr(RA, 'tree') and RA.tree is not None and not force_rebuild:
        print("[INFO] RAPTOR tree already exists.")
        return 0
    
    print(f"[INFO] Building RAPTOR tree with {max_docs} documents...")
    pubqa = load_dataset("pubmed_qa", "pqa_labeled")
    ds = pubqa["train"]
    
    docs = []
    for i in range(min(max_docs, len(ds))):
        context = normalize_context(ds[i].get("context", ""))
        if context and context.strip() and len(context) > 50:
            docs.append(context)
    
    if not docs:
        print("[WARN] No valid documents.")
        return 0
    
    print(f"[INFO] Adding {len(docs)} documents in batch...")
    separator = "\n\n--- DOCUMENT SEPARATOR ---\n\n"
    combined_text = separator.join(docs)
    
    try:
        RA.add_documents(combined_text)
        print(f"[INFO] Successfully added {len(docs)} documents to RAPTOR tree")
    except Exception as e:
        print(f"[ERROR] Failed to build tree: {e}")
        traceback.print_exc()
        return 0
    
    try:
        if hasattr(RA, 'tree'):
            if hasattr(RA.tree, 'all_nodes'):
                print(f"[INFO] Tree has {len(RA.tree.all_nodes)} total nodes")
            if hasattr(RA.tree, 'leaf_nodes'):
                print(f"[INFO] Tree has {len(RA.tree.leaf_nodes)} leaf nodes")
    except:
        pass
    
    return len(docs)

def create_qa_chain():
    """Single-shot QA chain without verification"""
    llm = model_manager.get_langchain_llm()
    
    prompt = PromptTemplate.from_template(
        """<start_of_turn>user
Context: {context}

Question: {question}

Based strictly on the context above, answer yes, no, or maybe. Provide a brief explanation citing specific facts from the context. If the context lacks sufficient information, answer "maybe" and explain why.
<end_of_turn>
<start_of_turn>model
Answer:"""
    )
    
    chain = prompt | llm
    print("[INFO] QA chain (no NLI) created")
    return chain

def raptor_agent_no_nli(RA, question: str, top_k=10, num_passages=3) -> Dict[str, Any]:
    """
    Simple RAPTOR-based QA without NLI verification:
    1. Retrieve top_k nodes from RAPTOR
    2. Combine passages into evidence
    3. Single QA call (no verification/revision)
    """
    print(f"\n[INFO] Querying (no NLI): {question}")
    
    # Retrieval
    retrieved = []
    try:
        retrieved = RA.retrieve(question, top_k=top_k)
        print(f"[INFO] Retrieved {len(retrieved)} passages")
    except Exception as e:
        print(f"[WARN] Retrieval failed: {e}")
        traceback.print_exc()
    
    # Extract text
    retrieved_texts = []
    for item in retrieved:
        if isinstance(item, str):
            retrieved_texts.append(item)
        elif isinstance(item, dict):
            text = item.get("text") or item.get("content") or ""
            if text:
                retrieved_texts.append(str(text))
        elif hasattr(item, 'text'):
            retrieved_texts.append(item.text)
    
    if not retrieved_texts:
        return {
            "question": question,
            "final_answer": "No evidence available.",
            "evidence": "",
            "rounds": 0,
            "used_langchain": True
        }
    
    # Use multiple passages
    num_passages = min(num_passages, len(retrieved_texts))
    combined_texts = []
    for i in range(num_passages):
        combined_texts.append(f"[Passage {i+1}] {retrieved_texts[i]}")
    
    evidence = " ".join(combined_texts)[:5000]
    print(f"[INFO] Evidence length: {len(evidence)} chars (from {num_passages} passages)")
    
    # Single QA call - UPDATED to use invoke
    qa_chain = create_qa_chain()
    try:
        answer = qa_chain.invoke({"context": evidence, "question": question})
        answer = clean_gemma_output(answer)
    except Exception as e:
        print(f"[WARN] Generation failed: {e}")
        traceback.print_exc()
        answer = "Unable to generate answer."
    
    print(f"[ANSWER] {answer[:200]}...")
    
    return {
        "question": question,
        "final_answer": answer,
        "evidence": evidence[:500],
        "rounds": 1,
        "used_langchain": True
    }


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_metrics(predictions, ground_truths):
    em_scores = [exact_match_score(p, gt) for p, gt in zip(predictions, ground_truths)]
    f1_scores = [f1_score(p, gt) for p, gt in zip(predictions, ground_truths)]
    return {
        'exact_match': sum(em_scores) / len(em_scores) * 100 if em_scores else 0,
        'f1': sum(f1_scores) / len(f1_scores) * 100 if f1_scores else 0,
        'total': len(predictions)
    }

def extract_yes_no_maybe(answer):
    """Extract yes/no/maybe from model output"""
    answer_lower = answer.lower()
    
    # Check for explicit answers at the start
    if answer_lower.startswith('yes'):
        return 'yes'
    elif answer_lower.startswith('no'):
        return 'no'
    elif answer_lower.startswith('maybe'):
        return 'maybe'
    
    # Check in first 100 chars
    first_part = answer_lower[:100]
    
    if 'yes' in first_part and 'no' not in first_part:
        return 'yes'
    elif 'no' in first_part and 'yes' not in first_part:
        return 'no'
    elif 'maybe' in first_part or 'unclear' in first_part or 'insufficient' in first_part:
        return 'maybe'
    
    return 'maybe'

def evaluate_system_no_nli(RA, num_samples=10, max_docs=100):
    """Evaluation without NLI verification"""
    print("\n" + "="*80)
    print("EVALUATION: RAPTOR + GEMMA 2B (NO NLI LOOP)")
    print("="*80)
    
    t0 = time.time()
    build_raptor_index(RA, max_docs=max_docs, force_rebuild=False)
    print(f"[INFO] Index built in {time.time() - t0:.2f}s")
    
    pubqa = load_dataset("pubmed_qa", "pqa_labeled")
    predictions = []
    ground_truths = []
    results = []
    
    for i in range(min(num_samples, len(pubqa['train']))):
        example = pubqa['train'][i]
        question = example.get('question', '')
        gt = example.get('final_decision', 'maybe')
        
        if not question:
            continue
        
        print(f"\n[EVAL {i+1}/{num_samples}]")
        result = raptor_agent_no_nli(RA, question, top_k=10)
        
        pred = extract_yes_no_maybe(result['final_answer'])
        predictions.append(pred)
        ground_truths.append(gt)
        
        results.append({
            'question': question,
            'prediction': pred,
            'ground_truth': gt,
            'match': pred == gt,
            'rounds': result['rounds']
        })
        
        print(f"GT: {gt} | Pred: {pred} | Match: {pred == gt}")
    
    metrics = compute_metrics(predictions, ground_truths)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Exact Match: {metrics['exact_match']:.2f}%")
    print(f"F1 Score: {metrics['f1']:.2f}%")
    print(f"Total Samples: {metrics['total']}")
    print(f"Model: Gemma 2B-IT (Full Precision)")
    print(f"Orchestration: LangChain + RAPTOR (NO NLI)")
    print("="*80)
    
    return metrics, results

if __name__ == "__main__":
    try:
        print("\n" + "="*80)
        print("MED-RAPTOR: RAPTOR AGENT WITHOUT NLI LOOP")
        print("Updated with Modern LangChain API (LCEL)")
        print("="*80)
        
        # Initialize RAPTOR
        print("[INFO] Initializing RAPTOR with Gemma 2B + LangChain (no NLI)...")
        RAC = RetrievalAugmentationConfig(
            summarization_model=GemmaLangChainSummarizationModel(),
            qa_model=GemmaLangChainQAModel(),
            embedding_model=SBertEmbeddingModel()
        )
        RA = RetrievalAugmentation(config=RAC)
        
        # Single question demo
        print("\n" + "="*80)
        print("SINGLE QUESTION DEMO")
        print("="*80)
        
        build_raptor_index(RA, max_docs=50, force_rebuild=False)
        
        test_q = "Is HER2 immunoreactivity significantly associated with disease-specific overall survival?"
        result = raptor_agent_no_nli(RA, test_q, top_k=10)
        
        print("\n" + "="*80)
        print("RESULT")
        print("="*80)
        print(f"Question: {result['question']}")
        print(f"\nAnswer: {result['final_answer']}")
        print(f"\nEvidence (preview): {result['evidence']}")
        print(f"Rounds: {result['rounds']}")
        print(f"LangChain Used: {result['used_langchain']}")
        print("="*80)
        
        # Full evaluation
        print("\n[MODE] Full Evaluation")
        evaluate_system_no_nli(RA, num_samples=10, max_docs=100)
        
    except Exception:
        traceback.print_exc()
    finally:
        builtins.input = original_input