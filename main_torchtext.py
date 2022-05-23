# Fundamental & Base
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import math
import json
import random
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

# Preprocessing
import pandas as pd
from torchtext.data import Field, BucketIterator, TabularDataset
from sklearn.utils import shuffle

# Model
import model.asmdepictor.Models as Asmdepictor
from model.asmdepictor.Translator import Translator
from model.asmdepictor.Optim import ScheduledOptim

# Metrics
from random_seed import set_random_seed
from sklearn.metrics import precision_recall_fscore_support



def cal_performance(pred, pred_sentence, gold, gold_sentence, trg_pad_idx, smoothing=False):
    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    pred_sentence = pred_sentence.max(-1)[1]
    gold = gold.contiguous().view(-1)
    pad_mask = gold.ne(trg_pad_idx)
    eos_mask = gold.ne(text.vocab.stoi['<eos>'])

    new_mask = pad_mask & eos_mask

    y_pred = pred.masked_select(new_mask).to("cpu")
    y_test = gold.masked_select(new_mask).to("cpu")

    f1 = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

    jaccard = True
    if jaccard:
        unique_pred = [set(l) for l in pred_sentence.tolist()]
        unique_gold = [set(l) for l in gold_sentence.tolist()]

        n_correct = 0
        n_word = 0
        e_value = 0
        for i in range(0, len(unique_gold)):
            union = unique_gold[i]
            if trg_pad_idx in union:
                union.remove(trg_pad_idx)
                union.remove(2) # remove special token for eval
                union.remove(3) # remove special token for eval

            intersection = unique_pred[i].intersection(unique_gold[i])
            if trg_pad_idx in intersection:
                intersection.remove(trg_pad_idx)
                intersection.remove(2) # remove special token for eval
                intersection.remove(3) # remove special token for eval
            
            alpha = 14
            if len(unique_pred[i]) <= len(unique_gold[i]) + alpha:
                e_value = 1
            else:
                e_value = math.exp(1-(len(unique_pred[i])/(len(unique_gold[i])+alpha))) 

            n_correct += len(intersection)*e_value
            n_word += len(union)

    else:
        n_correct = pred.eq(gold).masked_select(new_mask).sum().item() 
        n_word = new_mask.sum().item()

    return loss, n_correct, n_word, f1

def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    gold = gold.contiguous().view(-1)
    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        pad_mask = gold.ne(trg_pad_idx)
        
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss

def train(input_tensor, target_tensor, model, model_optimizer, smoothing, trg_pad_idx):
    total_loss, n_word_total, n_word_correct = 0, 0, 0 

    model_optimizer.zero_grad()
    target_tensor = target_tensor.transpose(0, 1)
    input_tensor = input_tensor.transpose(0, 1)

    gold = target_tensor[:, 1:].contiguous().view(-1)
    target_tensor = target_tensor[:, :-1]

    dec_output = model(input_tensor, target_tensor)
    dec_output_sentence = dec_output
    dec_output = dec_output.view(-1, dec_output.size(2))

    # backward and update parameters
    loss, n_correct, n_word, f1 = cal_performance(
        dec_output, dec_output_sentence, gold, target_tensor, trg_pad_idx, smoothing=smoothing)

    loss.backward()
    model_optimizer.step_and_update_lr()

    return loss.item(), n_correct, n_word, f1

def validate(input_tensor, target_tensor, model, trg_pad_idx):
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    target_tensor = target_tensor.transpose(0, 1)
    input_tensor = input_tensor.transpose(0, 1)

    gold = target_tensor[:, 1:].contiguous().view(-1)
    target_tensor = target_tensor[:, :-1]
    dec_output = model(input_tensor, target_tensor)
    dec_output_sentence = dec_output
    dec_output = dec_output.view(-1, dec_output.size(2))

    loss, n_correct, n_word, f1 = cal_performance(
        dec_output, dec_output_sentence, gold, target_tensor, trg_pad_idx, smoothing=False) 

    return loss.item(), n_correct, n_word, f1

def trainIters(model, n_epoch, train_iterator, test_iterator, model_optimizer, smoothing, trg_pad_idx):
    for i in range(1, n_epoch + 1):
        total_train_f1 = 0
        total_train_prec = 0
        total_train_rec = 0
        n_word_total = 0
        n_word_correct = 0
        total_loss = 0
        # one epoch
        print("\n", i, " Epoch...")
        print("\nTraining loop")
        model.module.train()
        for batch in tqdm(train_iterator):
            input_tensor = batch.code
            target_tensor = batch.text
            train_loss, n_correct, n_word, f1 = train(input_tensor, target_tensor, model, model_optimizer, smoothing, trg_pad_idx)
            total_train_prec += f1[0]
            total_train_rec += f1[1]
            total_train_f1 += f1[2]
            
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += train_loss

        loss_per_word = total_loss/n_word_total
        accuracy = n_word_correct/n_word_total

        lr = model_optimizer._optimizer.param_groups[0]['lr']
        writer.add_scalar('training loss', loss_per_word, i)
        writer.add_scalar('training Jaccard*', accuracy, i)
        writer.add_scalar('learning_rate', lr, i)

        mean_train_prec = total_train_prec/len(train_iterator)
        mean_train_rec = total_train_rec/len(train_iterator)
        mean_train_f1 = total_train_f1/len(train_iterator)

        print('loss : ', loss_per_word, 'Jaccard* : ', accuracy, 'F1 : ', mean_train_f1, 'Precision : ', mean_train_prec, 'Recall : ', mean_train_rec)

        # Validation loop
        print("\nValidation loop")
        model.module.eval()
        total_valid_f1 = 0
        total_valid_prec = 0
        total_valid_rec = 0
        n_word_total = 0
        n_word_correct = 0
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(test_iterator):
                input_tensor = batch.code
                target_tensor = batch.text
                target_loss, n_correct, n_word, f1 = validate(input_tensor, target_tensor, model, trg_pad_idx)
                total_valid_prec += f1[0]
                total_valid_rec += f1[1]
                total_valid_f1 += f1[2]

                n_word_total += n_word
                n_word_correct += n_correct
                total_loss += target_loss

            loss_per_word = total_loss/n_word_total
            accuracy = n_word_correct/n_word_total

            writer.add_scalar('validation loss', loss_per_word, i)
            writer.add_scalar('validation Jaccard*', accuracy, i)

            mean_valid_f1 = total_valid_f1/len(test_iterator)
            mean_valid_rec = total_valid_rec/len(test_iterator)
            mean_valid_prec = total_valid_prec/len(test_iterator)

            writer.add_scalar('F1-score', mean_valid_f1, i)
            writer.add_scalar('Precision', mean_valid_prec, i)
            writer.add_scalar('Recall', mean_valid_rec, i)

            print('loss : ', loss_per_word, 'Jaccard* : ', accuracy, 'F1 : ', mean_valid_f1, 'Precision : ', mean_valid_prec, 'Recall : ', mean_valid_rec)
            torch.save(model.state_dict(), str(i)+"_"+str(round(mean_valid_f1, 4))+"_asmdepictor.param")

            # Random select data in train and check training
            src_rand_train, tgt_rand_train = random_choice_from_train()
            train_hypothesis = make_a_hypothesis_transformer(model, src_rand_train, tgt_rand_train)
            print("Expected output : ", tgt_rand_train)
            print("Hypothesis output : ", "".join(train_hypothesis))
        

def sentence_to_tensor(sentence, model_type, src_or_tgt):
    sentence = tokenize(sentence)
    unk_idx = code.vocab.stoi[code.unk_token]
    pad_idx = code.vocab.stoi[code.pad_token]
    sentence_idx = [code.vocab.stoi.get(i, unk_idx) for i in sentence]

    for i in range(opt.max_token_seq_len-len(sentence_idx)):
        sentence_idx.append(code.vocab.stoi.get(i, pad_idx))

    sentence_tensor = torch.tensor(sentence_idx).to(device)
    sentence_tensor = sentence_tensor.unsqueeze(0)
    return sentence_tensor

def make_a_hypothesis_transformer(model, src, tgt):
    input_tensor = sentence_to_tensor(src, 'transformer', '')
    target_tensor = sentence_to_tensor(tgt, 'transformer', '')
    translator = Translator(
        model=model,
        beam_size=5,
        max_seq_len=opt.max_token_seq_len+3,
        src_pad_idx=code.vocab.stoi['<pad>'],
        trg_pad_idx=text.vocab.stoi['<pad>'],
        trg_bos_idx=text.vocab.stoi['<sos>'],
        trg_eos_idx=text.vocab.stoi['<eos>']).to(device)

    output_tensor = translator.translate_sentence(input_tensor)
    predict_sentence = ' '.join(text.vocab.itos[idx] for idx in output_tensor)
    predict_sentence = predict_sentence.replace('<sos>', '').replace('<eos>', '')
    return predict_sentence

def make_hypothesis_reference(model, test_src, test_tgt, model_type):
    hypothesis_list = list()
    reference_list = test_tgt
    print("Building hypothesis list...")
    for src, tgt in tqdm(zip(test_src, test_tgt)):
        hypothesis = make_a_hypothesis_transformer(model, src, tgt)
        hypothesis_list.append(hypothesis)
    return hypothesis_list, reference_list

def random_choice_from_train():
    # read json files made from torchtext
    train_data = list()
    for line in open("./dataset/valid.json", mode='r', encoding='utf-8'):
        train_data.append(json.loads(line))

    train_data = random.choice(train_data)
    source = train_data['Code'].lower()
    target = train_data['Text'].lower()

    return source, target

def preprocessing(src_file, tgt_file, max_token_seq_len):
    src_data = open(src_file, encoding='utf-8').read().split('\n')
    tgt_data = open(tgt_file, encoding='utf-8').read().split('\n')

    src_text_tok = [line.split() for line in src_data]
    src_tok_concat = [" ".join(tok[0:max_token_seq_len]) for tok in src_text_tok]

    tgt_text_tok = [line.split() for line in tgt_data]
    tgt_tok_concat = [" ".join(tok[0:max_token_seq_len]) for tok in tgt_text_tok]    

    raw_data = {'Code': [line for line in src_tok_concat],
                'Text': [line for line in tgt_tok_concat]}
    
    df = pd.DataFrame(raw_data, columns=['Code', 'Text'])

    return shuffle(df)

def main():
    parser = argparse.ArgumentParser()

    #parser.add_argument('-train_path', default=None, required=True)
    #parser.add_argument('-val_path', default=None, required=True)
    #parser.add_argument('-test_path', default=None, required=True)

    parser.add_argument('-batch_size', type=int, default=90)
    parser.add_argument('-epoch', type=int, default=150)
    parser.add_argument('-lr_mul', type=float, default=1.0)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-d_model', type=int, default=2048)
    parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-embs_share_weight', type=bool, default=True)
    parser.add_argument('-max_token_seq_len', type=int, default=300)
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=3)
    parser.add_argument('-proj_share_weight', type=bool, default=True)
    parser.add_argument('-scale_emb_or_prj', type=str, default='emb')
    parser.add_argument('-n_warmup_steps', type=int, default=18000)
    parser.add_argument('-smoothing', type=bool, default=True)
    #parser.add_argument('-output_path', default=None, required=True)
    parser.add_argument('-seed', type=int, default=123)
    parser.add_argument('-tensorboard_path', default='./runs/asmdepictor')

    global opt
    opt = parser.parse_args()

    # Cuda setting
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(opt.seed)

    # Hyperparameters
    #n_epoch = 150
    #batch_size = 90
    #d_inner_hid = 2048
    #d_k = 64 
    #d_model = 512 
    #d_v = 64
    #d_word_vec = 512
    #dropout = 0.1
    #embs_share_weight = True
    #global max_token_seq_len
    #max_token_seq_len = 300
    #n_head = 8 
    #n_layers = 3
    #proj_share_weight = True
    #scale_emb_or_prj = 'emb'

    #lr_mul = 1.0
    #n_warmup_steps = 18000

    #smoothing = True

    global writer
    writer = SummaryWriter(opt.tensorboard_path)

    train_src_dir = "./dataset/train_source.txt"
    valid_src_dir = "./dataset/test_source.txt"
    test_src_dir = "./dataset/test_source.txt"

    train_tgt_dir = "./dataset/train_target.txt"
    valid_tgt_dir = "./dataset/test_target.txt"
    test_tgt_dir = "./dataset/test_target.txt"

    train_set = preprocessing(train_src_dir, train_tgt_dir, opt.max_token_seq_len)
    valid_set = preprocessing(valid_src_dir, valid_tgt_dir, opt.max_token_seq_len)
    test_set = preprocessing(test_src_dir, test_tgt_dir, opt.max_token_seq_len)

    train_set.to_json('dataset/train.json', orient='records', lines=True)
    valid_set.to_json('dataset/valid.json', orient='records', lines=True)
    test_set.to_json('dataset/test.json', orient='records', lines=True)

    ## tokenize
    global tokenize
    global code
    global text
    tokenize = lambda x : x.split()

    code = Field(sequential=True, 
                use_vocab=True, 
                tokenize=tokenize, 
                lower=True,
                pad_token='<pad>',
                fix_length=opt.max_token_seq_len)

    text = Field(sequential=True, 
                use_vocab=True, 
                tokenize=tokenize, 
                lower=True,
                init_token='<sos>',
                eos_token='<eos>',
                pad_token='<pad>',
                fix_length=opt.max_token_seq_len)

    fields = {'Code' : ('code', code), 'Text' : ('text', text)}

    train_data, valid_data, test_data = TabularDataset.splits(path='',
                                                train='./dataset/train.json',
                                                test='./dataset/test.json',
                                                validation='./dataset/valid.json',
                                                format='json',
                                                fields=fields)

    code.build_vocab(train_data.code, train_data.code, min_freq=2)
    text.build_vocab(train_data.code, train_data.text, min_freq=0)
    
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=opt.batch_size,
        device = device,
        sort=False
    )

    src_pad_idx=code.vocab.stoi['<pad>']
    src_vocab_size=len(code.vocab.stoi)
    trg_pad_idx=text.vocab.stoi['<pad>']
    trg_vocab_size=len(text.vocab.stoi)

    model = Asmdepictor.Asmdepictor(src_vocab_size,
                                    trg_vocab_size,
                                    src_pad_idx=src_pad_idx,
                                    trg_pad_idx=trg_pad_idx,
                                    trg_emb_prj_weight_sharing=opt.proj_share_weight,
                                    emb_src_trg_weight_sharing=opt.embs_share_weight,
                                    d_k=opt.d_k,
                                    d_v=opt.d_v,
                                    d_model=opt.d_model,
                                    d_word_vec=opt.d_word_vec,
                                    d_inner=opt.d_inner_hid,
                                    n_layers=opt.n_layers,
                                    n_head=opt.n_head,
                                    dropout=opt.dropout,
                                    scale_emb_or_prj=opt.scale_emb_or_prj,
                                    n_position=opt.max_token_seq_len+3).to(device)

    model = nn.DataParallel(model)

    model_optimizer = ScheduledOptim(
        optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        opt.lr_mul, opt.d_model, opt.n_warmup_steps)
    
    # Train
    trainIters(model, opt.n_epoch, train_iterator, valid_iterator, model_optimizer, opt.smoothing, trg_pad_idx)
    writer.close()

    # read json files made from torchtext
    test_data = list()
    for line in open("./dataset/test.json", mode='r', encoding='utf-8'):
        test_data.append(json.loads(line))

    test_src = list()
    test_tgt = list()
    for d in test_data:
        test_src.append(d['Code'].lower())
        test_tgt.append(d['Text'].lower())
        
    hypothesis_list, reference_list = make_hypothesis_reference(model, test_src, test_tgt, 'transformer')

    if not os.path.exists('./predicted_output/'):
        os.makedirs('./predicted_output/')

    with open('./predicted_output/prediction.txt', mode="w", encoding="UTF-8", errors='ignore') as out:
        for hypo in hypothesis_list:
            out.write(hypo+'\n')

    with open('./predicted_output/ground_truth.txt', mode="w", encoding="UTF-8", errors='ignore') as out:
        for refer in reference_list:
            out.write(refer+'\n')

if __name__ == '__main__':
    main()