from sklearn.metrics import precision_recall_fscore_support
from datasets import load_metric
import math
import copy

def f1_accuracy(gold, pred):
    gold_cpy = copy.deepcopy(gold)
    pred_cpy = copy.deepcopy(pred)

    prec_total = 0
    rec_total = 0
    f1_total = 0
    for g, p in zip(gold_cpy, pred_cpy):
        for _ in range(len(g)-len(p)):
            p.append('')
        for _ in range(len(p)-len(g)):
            g.append('')
        prec, rec, f1, _ = precision_recall_fscore_support(g, p, average='weighted', zero_division=0)
        prec_total += prec
        rec_total += rec
        f1_total += f1
    return prec_total/len(gold), rec_total/len(gold), f1_total/len(gold)

def jac_accuracy(gold_sentence, pred_sentence):
    unique_pred = [set(l) for l in pred_sentence]
    unique_gold = [set(l) for l in gold_sentence]
    n_correct = 0
    n_word = 0
    e_value = 0
    for i in range(0, len(unique_gold)):
        union = unique_gold[i]
        intersection = unique_pred[i].intersection(unique_gold[i])
        alpha = 14
        if len(unique_pred[i]) <= len(unique_gold[i]) + alpha:
            e_value = 1
        else:
            e_value = math.exp(1-(len(unique_pred[i])/(len(unique_gold[i])+alpha)))
        n_correct += len(intersection)*e_value
        n_word += len(union)
    
    return n_correct/n_word

if __name__ == "__main__":
    hypothesis = open('./prediction.txt', mode='r', encoding='utf-8').read().split('\n')
    reference = open('./ground_truth.txt', mode='r', encoding='utf-8').read().split('\n')
    reference = [i.split('\t') for i in reference]

    reference = [i[0] for i in reference]
    hypothesis = [i for i in hypothesis]

    metric = load_metric('rouge')    
    rouge = metric.compute(predictions=hypothesis, references=reference)
    jac_reference = [i.split(' ') for i in reference]
    jac_hypothesis = [i.split(' ') for i in hypothesis]

    print(rouge["rougeL"]) 
    print(f1_accuracy(jac_reference, jac_hypothesis))
    print(jac_accuracy(jac_reference, jac_hypothesis))