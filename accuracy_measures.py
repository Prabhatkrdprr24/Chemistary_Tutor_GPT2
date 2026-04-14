from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score

# Example data
references = [
    "H2O is formed by the combination of hydrogen and oxygen.",
    "NaCl is common table salt."
]

predictions = [
    "Water is formed from hydrogen and oxygen.",
    "NaCl is table salt."
]

# BLEU Score
smoothie = SmoothingFunction().method4
bleu_scores = [sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie) 
               for ref, pred in zip(references, predictions)]
avg_bleu = sum(bleu_scores) / len(bleu_scores)
print("Average BLEU:", avg_bleu)

# ROUGE-L Score
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
rouge_l_scores = [scorer.score(ref, pred)['rougeL'].fmeasure 
                  for ref, pred in zip(references, predictions)]
avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
print("Average ROUGE-L:", avg_rouge_l)

# Exact Match
exact_matches = [int(ref.strip().lower() == pred.strip().lower()) 
                 for ref, pred in zip(references, predictions)]
em_score = sum(exact_matches) / len(exact_matches)
print("Exact Match:", em_score)

# Token-level F1 Score
def token_f1(ref, pred):
    ref_tokens = set(ref.lower().split())
    pred_tokens = set(pred.lower().split())
    common = ref_tokens & pred_tokens
    if len(common) == 0:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

f1_scores = [token_f1(ref, pred) for ref, pred in zip(references, predictions)]
avg_f1 = sum(f1_scores) / len(f1_scores)
print("Average F1:", avg_f1)
