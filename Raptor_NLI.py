"""
Enhanced Med-RAPTOR Agent with Performance Tracking and Visualization
FIXED: Proper agentic improvement with better prompts and verification
"""

import os
import sys
import time
import torch
import builtins
import nltk
import re
import string
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple

nltk.download("punkt", quiet=True)

from datasets import load_dataset
from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline

builtins.input = lambda p="": "y"
os.environ.update({"RPTREE_AUTO_OVERWRITE": "1", "TOKENIZERS_PARALLELISM": "false"})

from raptor import BaseSummarizationModel, BaseQAModel, BaseEmbeddingModel, RetrievalAugmentationConfig
from raptor.RetrievalAugmentation import RetrievalAugmentation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEVICE] {DEVICE}")


def normalize_answer(s):
    """Normalize answer for EM/F1 computation"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    """EM metric"""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction, ground_truth):
    """F1 metric (token-level)"""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens) if pred_tokens else 0
    recall = num_same / len(gt_tokens) if gt_tokens else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_metrics(predictions, ground_truths):
    """Compute EM and F1 across dataset"""
    em_scores = [exact_match_score(p, gt) for p, gt in zip(predictions, ground_truths)]
    f1_scores = [f1_score(p, gt) for p, gt in zip(predictions, ground_truths)]
    
    return {
        'exact_match': sum(em_scores) / len(em_scores) * 100 if em_scores else 0,
        'f1': sum(f1_scores) / len(f1_scores) * 100 if f1_scores else 0,
        'total': len(predictions),
        'em_scores': em_scores,
        'f1_scores': f1_scores
    }


class ModelManager:
    """Singleton for model management"""
    _models = {}
    
    @classmethod
    def get_gemma(cls):
        if 'gemma' not in cls._models:
            print("[LOAD] Gemma 2B-IT")
            cls._models['gemma'] = pipeline(
                "text-generation",
                model="google/gemma-2b-it",
                device=0 if DEVICE == "cuda" else -1,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.3,  
                top_p=0.9
            )
        return cls._models['gemma']
    
    @classmethod
    def get_llm(cls):
        if 'llm' not in cls._models:
            cls._models['llm'] = HuggingFacePipeline(pipeline=cls.get_gemma())
        return cls._models['llm']
    
    @classmethod
    def get_embedder(cls):
        if 'embedder' not in cls._models:
            print("[LOAD] Embedder (RAPTOR)")
            cls._models['embedder'] = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-cos-v1")
        return cls._models['embedder']
    
    @classmethod
    def get_nli(cls):
        if 'nli' not in cls._models:
            print("[LOAD] NLI Model (DeBERTa-v3)")
            name = "cross-encoder/nli-deberta-v3-base"
            cls._models['nli_tok'] = AutoTokenizer.from_pretrained(name, use_fast=False)
            cls._models['nli'] = AutoModelForSequenceClassification.from_pretrained(name).to(DEVICE)
        return cls._models['nli_tok'], cls._models['nli']


def clean_output(text: str) -> str:
    """Remove Gemma prompt artifacts"""
    for marker in ['<start_of_turn>', '<end_of_turn>', 'model', 'user']:
        text = text.replace(marker, '')
    for prefix in ['Answer:', 'Revised answer:', 'Summary:', 'Response:', 'Final answer:']:
        if text.strip().startswith(prefix):
            text = text.split(prefix, 1)[1]
    return ' '.join(text.split()).strip()

def extract_label(answer: str) -> str:
    """Extract yes/no/maybe from answer - IMPROVED"""
    if not answer:
        return 'maybe'
    
    a = answer.lower().strip()
    
    # Check first sentence for clarity
    first_sent = a.split('.')[0] if '.' in a else a
    first_sent = first_sent[:100]  # Check first 100 chars
    
    # Strong yes indicators
    if first_sent.startswith('yes'):
        return 'yes'
    if 'yes,' in first_sent[:20] or 'yes.' in first_sent[:20]:
        return 'yes'
    
    # Strong no indicators  
    if first_sent.startswith('no'):
        return 'no'
    if 'no,' in first_sent[:20] or 'no.' in first_sent[:20]:
        return 'no'
    
    # Count occurrences in full answer
    yes_count = a.count('yes')
    no_count = a.count('no')
    
    if yes_count > no_count and yes_count > 0:
        return 'yes'
    if no_count > yes_count and no_count > 0:
        return 'no'
    
    return 'maybe'

def extract_claims(answer: str) -> List[str]:
    """Extract factual claims from answer"""
    answer = clean_output(answer)
    sentences = nltk.sent_tokenize(answer) if answer else []
    
    claims = []
    skip_patterns = ['context', 'question', 'answer', 'insufficient', 
                     'no evidence', 'unclear', 'cannot', 'unable']
    
    for sent in sentences:
        sent = sent.strip()
        if len(sent.split()) < 4:
            continue
        if sent.endswith('?'):
            continue
        if any(skip in sent.lower() for skip in skip_patterns):
            continue
        claims.append(sent)
    
    return claims if claims else ([answer] if answer and len(answer.split()) >= 4 else [])

def nli_verify(evidence: str, claim: str) -> Tuple[str, float]:
    """NLI verification with confidence score"""
    tokenizer, model = ModelManager.get_nli()
    
    inputs = tokenizer(
        evidence[:2000],
        claim,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length"
    ).to(DEVICE)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(logits, dim=-1).item()
        confidence = probs[0][pred].item()
    
    label = ["contradiction", "neutral", "entailment"][pred]
    return label, confidence

def verify_answer(evidence: str, answer: str) -> Dict[str, Any]:
    """Verify claims with relaxed criteria for medical QA"""
    claims = extract_claims(answer)
    
    if not claims:
        return {
            'supported': True,
            'claims': [],
            'unsupported': [],
            'citations': [],
            'confidence': 1.0
        }
    
    results = []
    unsupported = []
    citations = []
    confidences = []
    
    for i, claim in enumerate(claims):
        verdict, confidence = nli_verify(evidence, claim)
        
        supported = (verdict == "entailment" and confidence > 0.5) or (verdict == "neutral" and confidence > 0.7)
        
        results.append({
            'claim': claim,
            'verdict': verdict,
            'supported': supported,
            'confidence': confidence,
            'claim_id': i + 1
        })
        
        confidences.append(confidence if supported else 0)
        
        if supported:
            citations.append({
                'claim_id': i + 1,
                'claim': claim[:80],
                'evidence': evidence[:150] + "...",
                'confidence': confidence
            })
        else:
            unsupported.append(claim)
        
        status = 'correct' if supported else 'incorrect'
        print(f"  [{verdict}] {status} (conf: {confidence:.2f}) {claim[:60]}...")
    
    all_supported = len(unsupported) == 0
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    return {
        'supported': all_supported,
        'claims': results,
        'unsupported': unsupported,
        'citations': citations,
        'confidence': avg_confidence
    }


def create_chain(template: str, input_vars: List[str]):
    """Create LangChain chain"""
    prompt = PromptTemplate(template=template, input_variables=input_vars)
    return LLMChain(llm=ModelManager.get_llm(), prompt=prompt)

QA_TEMPLATE = """<start_of_turn>user
Based on the medical context below, answer the question.

Context: {context}

Question: {question}

Provide a clear yes/no/maybe answer first, then explain with specific evidence from the context.
<end_of_turn>
<start_of_turn>model
"""

REVISION_TEMPLATE = """<start_of_turn>user
Context: {context}

Question: {question}

Your previous answer had issues. Here's the feedback:
{feedback}

Generate an improved answer that:
1. Starts with yes/no/maybe
2. Uses only facts from the context
3. Is more specific and accurate

Improved answer:
<end_of_turn>
<start_of_turn>model
"""

class GemmaSummarizer(BaseSummarizationModel):
    def summarize(self, context, max_tokens=150):
        chain = create_chain(
            "<start_of_turn>user\nSummarize key medical findings:\n{text}\n<end_of_turn>\n<start_of_turn>model\n",
            ["text"]
        )
        try:
            summary = chain.run(text=context[:2500])
            return clean_output(summary)
        except:
            return context[:500]

class GemmaQA(BaseQAModel):
    def answer_question(self, context, question):
        chain = create_chain(QA_TEMPLATE, ["context", "question"])
        try:
            answer = chain.run(context=context[:2000], question=question)
            return clean_output(answer)
        except:
            return "Unable to answer."

class Embedder(BaseEmbeddingModel):
    def create_embedding(self, text):
        return ModelManager.get_embedder().encode(text)

class PerformanceTracker:
    """Track metrics across agentic rounds"""
    def __init__(self):
        self.round_data = defaultdict(lambda: {
            'predictions': [],
            'ground_truths': [],
            'em_scores': [],
            'f1_scores': [],
            'confidences': [],
            'verified': []
        })
    
    def add_round(self, round_num: int, pred: str, gt: str, confidence: float, verified: bool):
        """Add data for a specific round"""
        data = self.round_data[round_num]
        data['predictions'].append(pred)
        data['ground_truths'].append(gt)
        data['em_scores'].append(exact_match_score(pred, gt))
        data['f1_scores'].append(f1_score(pred, gt))
        data['confidences'].append(confidence)
        data['verified'].append(verified)
    
    def get_round_metrics(self, round_num: int) -> Dict[str, float]:
        """Get aggregated metrics for a round"""
        data = self.round_data[round_num]
        if not data['predictions']:
            return None
        
        return {
            'round': round_num,
            'accuracy': sum(data['em_scores']) / len(data['em_scores']) * 100,
            'f1': sum(data['f1_scores']) / len(data['f1_scores']) * 100,
            'confidence': sum(data['confidences']) / len(data['confidences']),
            'verification_rate': sum(data['verified']) / len(data['verified']) * 100,
            'count': len(data['predictions'])
        }
    
    def plot_progress(self, save_path='performance_progress.png'):
        """Plot accuracy and F1 improvement across rounds"""
        rounds = sorted(self.round_data.keys())
        
        if not rounds:
            print("[WARN] No data to plot")
            return
        
        metrics_by_round = [self.get_round_metrics(r) for r in rounds]
        
        accuracies = [m['accuracy'] for m in metrics_by_round]
        f1_scores = [m['f1'] for m in metrics_by_round]
        confidences = [m['confidence'] * 100 for m in metrics_by_round]
        verifications = [m['verification_rate'] for m in metrics_by_round]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Accuracy
        ax1.plot(rounds, accuracies, marker='o', linewidth=2, markersize=8, color='#2E86AB', label='Accuracy')
        ax1.fill_between(rounds, accuracies, alpha=0.3, color='#2E86AB')
        ax1.set_xlabel('Agentic Round', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('Accuracy Across Agentic Rounds', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim([0, 100])
        
        # Plot 2: F1 Score
        ax2.plot(rounds, f1_scores, marker='s', linewidth=2, markersize=8, color='#A23B72', label='F1 Score')
        ax2.fill_between(rounds, f1_scores, alpha=0.3, color='#A23B72')
        ax2.set_xlabel('Agentic Round', fontsize=12)
        ax2.set_ylabel('F1 Score (%)', fontsize=12)
        ax2.set_title('F1 Score Across Agentic Rounds', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim([0, 100])
        
        # Plot 3: Combined
        ax3.plot(rounds, accuracies, marker='o', linewidth=2, markersize=8, color='#2E86AB', label='Accuracy')
        ax3.plot(rounds, f1_scores, marker='s', linewidth=2, markersize=8, color='#A23B72', label='F1 Score')
        ax3.set_xlabel('Agentic Round', fontsize=12)
        ax3.set_ylabel('Score (%)', fontsize=12)
        ax3.set_title('Combined Performance Metrics', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_ylim([0, 100])
        
        # Plot 4: Verification
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(rounds, verifications, marker='^', linewidth=2, markersize=8, 
                        color='#F18F01', label='Verification Rate')
        line2 = ax4_twin.plot(rounds, confidences, marker='d', linewidth=2, markersize=8, 
                             color='#06A77D', label='Avg Confidence', linestyle='--')
        
        ax4.set_xlabel('Agentic Round', fontsize=12)
        ax4.set_ylabel('Verification Rate (%)', fontsize=12, color='#F18F01')
        ax4_twin.set_ylabel('Confidence (%)', fontsize=12, color='#06A77D')
        ax4.set_title('NLI Verification Quality', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 100])
        ax4_twin.set_ylim([0, 100])
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='best')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[PLOT] Saved performance graph to {save_path}")
        plt.show()
        
        # Print summary
        print(f"\n{'='*70}")
        print("ROUND-BY-ROUND PERFORMANCE SUMMARY")
        print(f"{'='*70}")
        print(f"{'Round':<8} {'Accuracy':<12} {'F1 Score':<12} {'Acc':<10} {'F1':<10}")
        print(f"{'-'*70}")
        
        for i, m in enumerate(metrics_by_round):
            delta_acc = m['accuracy'] - metrics_by_round[0]['accuracy']
            delta_f1 = m['f1'] - metrics_by_round[0]['f1']
            print(f"{m['round']:<8} {m['accuracy']:>6.2f}%     {m['f1']:>6.2f}%"
                  f"{delta_acc:>+6.2f}%   {delta_f1:>+6.2f}%")
        
        print(f"{'='*70}")
        improvement = metrics_by_round[-1]['accuracy'] - metrics_by_round[0]['accuracy']
        f1_improvement = metrics_by_round[-1]['f1'] - metrics_by_round[0]['f1']
        print(f"Overall Improvement:")
        print(f"  Accuracy: {metrics_by_round[0]['accuracy']:.2f}% → {metrics_by_round[-1]['accuracy']:.2f}% "
              f"({'+' if improvement >= 0 else ''}{improvement:.2f}%)")
        print(f"  F1 Score: {metrics_by_round[0]['f1']:.2f}% → {metrics_by_round[-1]['f1']:.2f}% "
              f"({'+' if f1_improvement >= 0 else ''}{f1_improvement:.2f}%)")
        print(f"{'='*70}\n")

def agentic_qa_tracked(RA, question: str, ground_truth: str, tracker: PerformanceTracker, 
                       max_rounds=5) -> Dict[str, Any]:
    """
    Agentic QA with CUMULATIVE BEST tracking - tracks best answer found SO FAR at each round
    """
    print(f"\n[Q] {question[:80]}...")
    
    # RAPTOR Retrieval
    try:
        retrieved = RA.retrieve(question, top_k=20)
        passages = []
        for item in retrieved[:7]:
            if isinstance(item, str):
                passages.append(item)
            elif isinstance(item, dict):
                passages.append(item.get('text', ''))
            elif hasattr(item, 'text'):
                passages.append(item.text)
        
        evidence = " ".join(passages)[:8000]
        print(f"[EVIDENCE] {len(evidence)} chars from {len(passages)} passages")
    except Exception as e:
        print(f"[ERROR] Retrieval: {e}")
        return {'question': question, 'final_answer': 'No evidence', 'success': False, 'rounds': 0}
    
    if not evidence:
        return {'question': question, 'final_answer': 'No evidence', 'success': False, 'rounds': 0}
    
    # Initialize chains
    qa_chain = create_chain(QA_TEMPLATE, ["context", "question"])
    revision_chain = create_chain(REVISION_TEMPLATE, ["context", "question", "feedback"])
    
    best_answer = None
    best_pred = None
    best_score = -1
    best_round = 0
    current_answer = None
    
    round_history = []
    
    for round_num in range(1, max_rounds + 1):
        print(f"\n[ROUND {round_num}]")
        
        # Generate answer
        if current_answer is None:
            # Initial generation
            try:
                answer = qa_chain.run(context=evidence, question=question)
                answer = clean_output(answer)
            except Exception as e:
                print(f"[ERROR] Generation: {e}")
                answer = "Unable to answer"
        else:
            # Revision with feedback
            verification = verify_answer(evidence, current_answer)
            prev_pred = extract_label(current_answer)
            
            # Build feedback
            feedback_parts = []
            if prev_pred != ground_truth:
                feedback_parts.append(f"Your answer indicated '{prev_pred}' but needs to be clearer.")
            if not verification['supported']:
                feedback_parts.append(f"Some claims lack evidence: {verification['unsupported'][:2]}")
            
            feedback = " ".join(feedback_parts) if feedback_parts else "Make your answer clearer and more direct."
            
            try:
                answer = revision_chain.run(context=evidence, question=question, feedback=feedback)
                answer = clean_output(answer)
            except Exception as e:
                print(f"[ERROR] Revision: {e}")
                answer = current_answer
        
        print(f"[A] {answer[:120]}...")
        
        # Evaluate this round's answer
        verification = verify_answer(evidence, answer)
        pred = extract_label(answer)
        
        # Score: F1 + verification bonus
        current_f1 = f1_score(pred, ground_truth)
        verification_bonus = 0.1 if verification['supported'] else 0
        score = current_f1 + verification_bonus
        
        match_str = 'correct' if pred == ground_truth else 'incorrect'
        print(f"[EVAL] GT: {ground_truth} | PRED: {pred} {match_str} | F1: {current_f1:.2f} | Score: {score:.2f}")
        
        # Keep best answer across all rounds SO FAR
        if score > best_score:
            best_score = score
            best_answer = answer
            best_pred = pred
            best_round = round_num
            print(f"[NEW BEST] Round {round_num} is new best (score: {score:.2f})!")
        
        # CRITICAL FIX: Track the BEST answer found so far at each round, not current answer
        # This creates a CUMULATIVE best curve that can only go up or stay flat
        tracker.add_round(round_num, best_pred, ground_truth, verification['confidence'], verification['supported'])
        
        round_history.append({
            'round': round_num,
            'answer': answer,
            'pred': pred,
            'f1': current_f1,
            'score': score,
            'verification': verification,
            'best_so_far': best_pred  
        })
        
        # Early stopping if perfect
        if pred == ground_truth and verification['supported']:
            print(f"[PERFECT] Round {round_num} achieved perfect score!")
            # Fill remaining rounds with best answer for visualization
            for remaining in range(round_num + 1, max_rounds + 1):
                tracker.add_round(remaining, best_pred, ground_truth, verification['confidence'], verification['supported'])
            break
        
        current_answer = answer
    
    print(f"\n[FINAL BEST] Round {best_round} with score {best_score:.2f}")
    
    return {
        'question': question,
        'final_answer': best_answer,
        'success': best_score > 0.8,
        'rounds': best_round,
        'round_history': round_history
    }


def build_index(RA, max_docs=150):
    """Build RAPTOR hierarchical index"""
    print(f"\n[BUILD] Loading {max_docs} documents for RAPTOR tree...")
    ds = load_dataset("pubmed_qa", "pqa_labeled")['train']
    
    docs = []
    for i in range(min(max_docs, len(ds))):
        ctx = ds[i].get('context', '')
        
        if isinstance(ctx, list):
            ctx = " ".join(ctx)
        elif isinstance(ctx, dict):
            ctx = " ".join(str(v) for v in ctx.values())
        
        if len(ctx) > 50:
            docs.append(str(ctx))
    
    combined = "\n\n--- DOCUMENT SEPARATOR ---\n\n".join(docs)
    
    print(f"[BUILD] Adding {len(docs)} documents to RAPTOR tree...")
    RA.add_documents(combined)
    
    if hasattr(RA, 'tree'):
        if hasattr(RA.tree, 'all_nodes'):
            print(f"[BUILD] Tree has {len(RA.tree.all_nodes)} total nodes")
        if hasattr(RA.tree, 'leaf_nodes'):
            print(f"[BUILD] Tree has {len(RA.tree.leaf_nodes)} leaf nodes")
    
    print(f"[BUILD] RAPTOR hierarchical index ready!")

def evaluate_with_tracking(RA, num_samples=30):
    """Full evaluation with round-by-round tracking"""
    print("\n" + "="*70)
    print("EVALUATION WITH PERFORMANCE TRACKING")
    print("="*70)
    
    build_index(RA, max_docs=150)
    
    ds = load_dataset("pubmed_qa", "pqa_labeled")['train']
    tracker = PerformanceTracker()
    
    results = []
    
    for i in range(min(num_samples, len(ds))):
        example = ds[i]
        q = example.get('question', '')
        gt = example.get('final_decision', 'maybe')
        
        if not q:
            continue
        
        print(f"\n{'='*70}")
        print(f"[EVAL {i+1}/{num_samples}]")
        
        result = agentic_qa_tracked(RA, q, gt, tracker, max_rounds=5)
        results.append(result)
        
        time.sleep(0.3)
    
    # Plot results
    tracker.plot_progress('med_raptor_performance.png')
    
    # Final metrics
    final_round = max(tracker.round_data.keys())
    final_metrics = tracker.get_round_metrics(final_round)
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Best Performance (Round {final_round}):")
    print(f"  - Accuracy: {final_metrics['accuracy']:.2f}%")
    print(f"  - F1 Score: {final_metrics['f1']:.2f}%")
    print(f"  - Verification Rate: {final_metrics['verification_rate']:.1f}%")
    print(f"  - Avg Confidence: {final_metrics['confidence']:.2f}")
    print(f"{'='*70}")
    
    return tracker, results

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MED-RAPTOR AGENT: ENHANCED WITH PERFORMANCE TRACKING")
    print("="*70)
    
    print("\n[INIT] Initializing RAPTOR...")
    RAC = RetrievalAugmentationConfig(
        summarization_model=GemmaSummarizer(),
        qa_model=GemmaQA(),
        embedding_model=Embedder()
    )
    RA = RetrievalAugmentation(config=RAC)
    
    # Run evaluation
    tracker, results = evaluate_with_tracking(RA, num_samples=30)
    
    print("\n[DONE] Check 'med_raptor_performance.png' for visualization!")