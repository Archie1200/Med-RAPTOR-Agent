"""
Simple RAG System for PubMedQA Dataset with Evaluation
Uses: sentence-transformers + ChromaDB + Gemma 2B (Local)
No external API needed - all local processing
"""

import os
import re
import string
import time
from typing import List, Dict, Any
from collections import Counter
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

print("[INFO] All dependencies imported successfully")

def normalize_answer(s):
    """Normalize answer for comparison"""
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
    """Calculate exact match score"""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction, ground_truth):
    """Calculate F1 score"""
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
    """Compute evaluation metrics"""
    em_scores = [exact_match_score(p, gt) for p, gt in zip(predictions, ground_truths)]
    f1_scores = [f1_score(p, gt) for p, gt in zip(predictions, ground_truths)]
    return {
        'exact_match': sum(em_scores) / len(em_scores) * 100 if em_scores else 0,
        'f1': sum(f1_scores) / len(f1_scores) * 100 if f1_scores else 0,
        'total': len(predictions)
    }

def extract_yes_no_maybe(answer: str) -> str:
    """Extract yes/no/maybe from model output"""
    if not answer:
        return 'maybe'
    
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


class SimplePubMedRAG:
    """
    Simple RAG system for PubMedQA dataset
    - Embeddings: sentence-transformers
    - Vector DB: ChromaDB
    - LLM: Gemma 2B (Local, no API needed)
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        collection_name: str = "pubmedqa",
        max_docs: int = 1000
    ):
        """
        Initialize Simple RAG System
        
        Args:
            embedding_model: SentenceTransformer model (768-dim recommended)
            collection_name: ChromaDB collection name
            max_docs: Maximum documents to index
        """
        print("\n" + "="*80)
        print("INITIALIZING SIMPLE PUBMEDQA RAG SYSTEM (GEMMA 2B)")
        print("="*80)
        
        self.max_docs = max_docs
        
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n[INFO] Device: {self.device}")
        
        # 1. Load PubMedQA dataset
        print("\n[1/5] Loading PubMedQA dataset...")
        self.dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
        print(f"Loaded {len(self.dataset)} total records")
        print(f"Using first {max_docs} documents for indexing")
        
        # 2. Initialize embedding model
        print(f"\n[2/5] Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        embed_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {embed_dim}")
        
        # 3. Initialize ChromaDB
        print("\n[3/5] Setting up ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(path="./pubmedqa_chroma_db")
        
        # Delete existing collection if present
        try:
            self.chroma_client.delete_collection(name=collection_name)
            print("Cleared existing collection")
        except:
            pass
        
        self.collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"ChromaDB collection '{collection_name}' ready")
        
        # 4. Initialize Gemma 2B
        print("\n[4/5] Loading Gemma 2B model...")
        self.load_gemma()
        
        # 5. Index documents
        print("\n[5/5]  Indexing documents into vector database...")
        self.index_documents()
        print("\RAG SYSTEM READY!")
        print("="*80)
    
    
    def load_gemma(self):
        """Load Gemma 2B model locally"""
        model_name = "google/gemma-2b-it"
        
        try:
            print(f"   Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            print(f"   Loading model (this may take a minute)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Create pipeline
            self.llm_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            print(f"Gemma 2B loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading Gemma 2B: {e}")
            raise
    
    def clean_gemma_output(self, text: str) -> str:
        """Clean Gemma output from prompt artifacts"""
        if not text:
            return text
        
        # Remove model tags
        text = text.split('model')[-1] if 'model' in text else text
        text = text.replace('', '')
        text = text.replace('user', '')
        
        # Remove common prefixes
        prefixes = ['Answer:', 'Response:', 'Output:', 'Context:', 'Question:']
        for prefix in prefixes:
            if text.strip().startswith(prefix):
                text = text.split(prefix, 1)[1]
        
        return ' '.join(text.split()).strip()
    
    def prepare_document_text(self, example: Dict) -> str:
        """
        Convert PubMedQA example to searchable text
        
        Format: Question + Context + Answer
        """
        question = example.get('question', '')
        
        # Context can be dict with 'contexts' key or direct list
        context_data = example.get('context', {})
        if isinstance(context_data, dict):
            contexts = context_data.get('contexts', [])
        else:
            contexts = context_data if isinstance(context_data, list) else []
        
        # Join contexts
        context_text = ' '.join(contexts) if contexts else ''
        
        # Long answer
        long_answer = example.get('long_answer', '')
        
        # Final decision
        final_decision = example.get('final_decision', '')
        
        # Combine all
        doc_text = f"""Question: {question}

Context: {context_text}

Answer: {long_answer}

Decision: {final_decision}"""
        
        return doc_text
    
    def index_documents(self):
        """Index PubMedQA documents into ChromaDB"""
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        # Process dataset
        num_docs = min(self.max_docs, len(self.dataset))
        
        print(f"\Processing {num_docs} documents...")
        
        for i in range(num_docs):
            example = self.dataset[i]
            
            # Prepare document text
            doc_text = self.prepare_document_text(example)
            
            # Skip if empty
            if len(doc_text.strip()) < 50:
                continue
            
            # Generate embedding
            embedding = self.embedding_model.encode(doc_text, show_progress_bar=False)
            
            # Prepare metadata
            metadata = {
                'question': example.get('question', '')[:500],
                'long_answer': example.get('long_answer', '')[:1000],
                'final_decision': example.get('final_decision', ''),
                'pubid': str(example.get('pubid', i))
            }
            
            documents.append(doc_text)
            embeddings.append(embedding.tolist())
            metadatas.append(metadata)
            ids.append(f"doc_{i}")
            
            # Progress
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{num_docs}...")
        
        # Add to ChromaDB in batches
        print(f"\ Adding {len(documents)} documents to ChromaDB...")
        batch_size = 100
        
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            
            self.collection.add(
                documents=documents[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
            
            print(f"  Indexed {end_idx}/{len(documents)}...")
        
        print(f"\Successfully indexed {len(documents)} documents")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top-k most relevant documents
        
        Args:
            query: User's question
            top_k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query, show_progress_bar=False)
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Format results
        retrieved_docs = []
        
        if results['ids']:
            for i in range(len(results['ids'][0])):
                doc = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                retrieved_docs.append(doc)
        
        return retrieved_docs
    
    def generate_answer(self, query: str, top_k: int = 3, verbose: bool = True) -> Dict[str, Any]:
        """
        Generate answer using RAG pipeline
        
        Args:
            query: User's question
            top_k: Number of documents to retrieve
            verbose: Print progress messages
            
        Returns:
            Dictionary with answer and sources
        """
        if verbose:
            print(f"\nProcessing query: {query}")
        
        # Step 1: Retrieve
        if verbose:
            print(f"   Retrieving top-{top_k} documents...")
        retrieved_docs = self.retrieve(query, top_k)
        
        if not retrieved_docs:
            return {
                'query': query,
                'answer': 'No relevant information found in the database.',
                'sources': [],
                'num_sources': 0
            }
        
        if verbose:
            print(f"Retrieved {len(retrieved_docs)} documents")
        
        # Step 2: Prepare context
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(
                f"[Source {i}]\n"
                f"Question: {doc['metadata']['question']}\n"
                f"Answer: {doc['metadata']['long_answer']}\n"
                f"Decision: {doc['metadata']['final_decision']}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Step 3: Create prompt for Gemma
        prompt = f"""You are a medical research assistant. Use the provided context to answer the question accurately.

Context from PubMed Research:
{context}

Question: {query}

Instructions:
- Answer based on the context provided
- Be concise and clear
- Use medical terminology appropriately
- If context is insufficient, state this

Answer:"""
        
        # Step 4: Generate with Gemma 2B
        if verbose:
            print(f"Generating answer with Gemma 2B...")
        
        try:
            outputs = self.llm_pipeline(
                prompt,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = outputs[0]['generated_text']
            
            # Remove the prompt from output
            if prompt in generated_text:
                answer = generated_text.replace(prompt, '').strip()
            else:
                answer = generated_text.strip()
            
            # Clean output
            answer = self.clean_gemma_output(answer)
            
            if verbose:
                print(f"Answer generated")
            
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
            if verbose:
                print(f"Error: {e}")
        
        # Step 5: Format response
        return {
            'query': query,
            'answer': answer,
            'sources': [
                {
                    'id': doc['id'],
                    'question': doc['metadata']['question'],
                    'answer': doc['metadata']['long_answer'][:200] + "...",
                    'decision': doc['metadata']['final_decision'],
                    'relevance_score': 1 - doc['distance'] if doc['distance'] else None
                }
                for doc in retrieved_docs
            ],
            'num_sources': len(retrieved_docs)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            'total_documents': self.collection.count(),
            'embedding_model': str(self.embedding_model),
            'embedding_dim': self.embedding_model.get_sentence_embedding_dimension(),
            'llm_model': 'Gemma 2B-IT (Local)',
            'device': self.device,
            'vector_db': 'ChromaDB'
        }
    
    def evaluate(self, num_samples: int = 100, top_k: int = 3) -> Dict[str, Any]:
        """
        Evaluate RAG system on PubMedQA dataset
        
        Args:
            num_samples: Number of samples to evaluate
            top_k: Number of documents to retrieve per query
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "="*80)
        print(f"EVALUATING BASELINE RAG ON {num_samples} SAMPLES")
        print("="*80)
        
        predictions = []
        ground_truths = []
        results = []
        
        start_time = time.time()
        
        for i in range(min(num_samples, len(self.dataset))):
            example = self.dataset[i]
            question = example.get('question', '')
            gt = example.get('final_decision', 'maybe')
            
            if not question:
                continue
            
            print(f"\n[{i+1}/{num_samples}] Evaluating: {question[:60]}...")
            
            # Generate answer
            result = self.generate_answer(query=question, top_k=top_k, verbose=False)
            
            # Extract prediction
            pred = extract_yes_no_maybe(result['answer'])
            
            predictions.append(pred)
            ground_truths.append(gt)
            
            match = pred == gt
            results.append({
                'question': question,
                'prediction': pred,
                'ground_truth': gt,
                'match': match,
                'answer': result['answer']
            })
            
            status = "Correct" if match else "Incorrect"
            print(f"   GT: {gt} | Pred: {pred} | {status}")
        
        elapsed_time = time.time() - start_time
        
        # Compute metrics
        metrics = compute_metrics(predictions, ground_truths)
        
        # Calculate accuracy
        accuracy = sum(r['match'] for r in results) / len(results) * 100 if results else 0
        
        # Print results
        print("\n" + "="*80)
        print("EVALUATION RESULTS - BASELINE RAG")
        print("="*80)
        print(f"Exact Match: {metrics['exact_match']:.2f}%")
        print(f"F1 Score: {metrics['f1']:.2f}%")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Total Samples: {metrics['total']}")
        print(f"Correct: {sum(r['match'] for r in results)}/{len(results)}")
        print(f"Time Elapsed: {elapsed_time:.2f}s")
        print(f"Avg Time/Sample: {elapsed_time/len(results):.2f}s")
        print(f"\nModel: Gemma 2B-IT (Local)")
        print(f"Embeddings: {self.embedding_model.get_sentence_embedding_dimension()}-dim")
        print(f"Top-K Retrieved: {top_k}")
        print(f"Device: {self.device}")
        print("="*80)
        
        return {
            'metrics': metrics,
            'accuracy': accuracy,
            'results': results,
            'time_elapsed': elapsed_time,
            'avg_time_per_sample': elapsed_time / len(results) if results else 0
        }


if __name__ == "__main__":
    
    # Initialize RAG system with Gemma 2B (no API key needed!)
    rag = SimplePubMedRAG(
        embedding_model="sentence-transformers/all-mpnet-base-v2",  # 768-dim
        max_docs=100  # Use 100 docs for evaluation (same as RAPTOR)
    )
    

    print("\n" + "="*80)
    print("RUNNING EVALUATION")
    print("="*80)
    
    eval_results = rag.evaluate(
        num_samples=10,  
        top_k=3          
    )
    
    # Display detailed results
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    
    correct = 0
    total = 0
    
    for i, result in enumerate(eval_results['results'][:10], 1):  
        status = "Correct" if result['match'] else "Incorrect"
        print(f"\n[{i}] {status} Q: {result['question'][:70]}...")
        print(f"    GT: {result['ground_truth']} | Pred: {result['prediction']}")
        if result['match']:
            correct += 1
        total += 1
    
    print(f"\n{'='*80}")
    print(f"Summary: {correct}/{total} correct ({correct/total*100:.1f}%)")
    print("="*80)
    
    
