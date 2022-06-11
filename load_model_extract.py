from torchtext.data import Field, BucketIterator, TabularDataset
from model.asmdepictor.Translator import Translator
from tqdm import tqdm
import model.asmdepictor.Models as Asmdepictor
import pandas as pd
import torch.nn as nn
import torch
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_token_seq_len = 300

def make_hypothesis_reference(model, test_src, test_tgt, model_type):
    hypothesis_list = list()
    reference_list = test_tgt
    print("Building hypothesis list...")
    for src, tgt in tqdm(zip(test_src, test_tgt)):
        if model_type == 'transformer':
            hypothesis = make_a_hypothesis_transformer(model, src, tgt)
        hypothesis_list.append(hypothesis)
    return hypothesis_list, reference_list

def make_a_hypothesis_transformer(model, src, tgt):
    input_tensor = sentence_to_tensor(src, 'transformer', '')
    target_tensor = sentence_to_tensor(tgt, 'transformer', '')
    translator = Translator(
        model=model,
        beam_size=5,
        max_seq_len=max_token_seq_len+3,
        src_pad_idx=code.vocab.stoi['<pad>'],
        trg_pad_idx=text.vocab.stoi['<pad>'],
        trg_bos_idx=text.vocab.stoi['<sos>'],
        trg_eos_idx=text.vocab.stoi['<eos>']).to(device)

    output_tensor = translator.translate_sentence(input_tensor)
    removed_sos_eos = output_tensor[1:-1].copy()
    predict_sentence = ' '.join(text.vocab.itos[idx] for idx in output_tensor)
    predict_sentence = predict_sentence.replace('<sos>', '').replace('<eos>', '')
    return predict_sentence

def sentence_to_tensor(sentence, model_type, src_or_tgt):
    if model_type == 'transformer':
        sentence = tokenize(sentence)
        unk_idx = code.vocab.stoi[code.unk_token]
        pad_idx = code.vocab.stoi[code.pad_token]
        sentence_idx = [code.vocab.stoi.get(i, unk_idx) for i in sentence]

        for i in range(max_token_seq_len-len(sentence_idx)):
            sentence_idx.append(code.vocab.stoi.get(i, pad_idx))

        sentence_tensor = torch.tensor(sentence_idx).to(device)
        sentence_tensor = sentence_tensor.unsqueeze(0)
        return sentence_tensor

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

    return df

if __name__ == "__main__":
    train_json_dir = './dataset/1_train.json'
    test_json_dir = './dataset/test_100.json'
    model_path = './dataset/asmdepictor_pretrained.param'

    global tokenize
    tokenize = lambda x : x.split()

    code = Field(sequential=True, 
                use_vocab=True, 
                tokenize=tokenize, 
                lower=True,
                pad_token='<pad>',
                fix_length=max_token_seq_len)

    text = Field(sequential=True, 
                use_vocab=True, 
                tokenize=tokenize, 
                lower=True,
                init_token='<sos>',
                eos_token='<eos>',
                pad_token='<pad>',
                fix_length=max_token_seq_len)

    fields = {'Code' : ('code', code), 'Text' : ('text', text)}

    train_data, test_data = TabularDataset.splits(path='',
                                                train=train_json_dir,
                                                test=test_json_dir,
                                                format='json',
                                                fields=fields)

    # share train & tgt word2idx
    code.build_vocab(train_data.code, train_data.text, min_freq=2)
    text.build_vocab(train_data.code, train_data.text, min_freq=0)

    train_iterator, test_iterator = BucketIterator.splits(
        (train_data, test_data),
        batch_size=90,
        device = "cuda",
        sort=False
    )

    # model
    batch_size=90
    d_inner_hid=2048
    d_k=64 
    d_model=512 
    d_v=64
    d_word_vec=512
    dropout=0.1
    embs_share_weight=True
    n_head=8 
    n_layers=3 
    proj_share_weight=True
    scale_emb_or_prj='emb'

    src_pad_idx=code.vocab.stoi['<pad>']
    src_vocab_size=len(code.vocab.stoi)
    trg_pad_idx=text.vocab.stoi['<pad>']
    trg_vocab_size=len(text.vocab.stoi)

    model = Asmdepictor.Asmdepictor(src_vocab_size,
                                    trg_vocab_size,
                                    src_pad_idx=src_pad_idx,
                                    trg_pad_idx=trg_pad_idx,
                                    trg_emb_prj_weight_sharing=proj_share_weight,
                                    emb_src_trg_weight_sharing=embs_share_weight,
                                    d_k=d_k,
                                    d_v=d_v,
                                    d_model=d_model,
                                    d_word_vec=d_word_vec,
                                    d_inner=d_inner_hid,
                                    n_layers=n_layers,
                                    n_head=n_head,
                                    dropout=dropout,
                                    scale_emb_or_prj=scale_emb_or_prj,
                                    n_position=max_token_seq_len+3).to(device)

    model = nn.DataParallel(model)
    state_dict = torch.load(model_path)

    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # read json files made from torchtext
    test_data = list()
    for line in open(test_json_dir, mode='r', encoding='utf-8'):
        test_data.append(json.loads(line))

    test_src = list()
    test_tgt = list()
    for d in test_data:
        test_src.append(d['Code'].lower())
        test_tgt.append(d['Text'].lower())

    hypothesis_list, reference_list = make_hypothesis_reference(model, test_src, test_tgt, 'transformer')
    hypothesis_list = [i[1:-1] for i in hypothesis_list]

    if not os.path.exists('./predicted_output/'):
        os.makedirs('./predicted_output/')

    with open("./predicted_output/prediction.txt", mode='w', encoding='utf-8') as out:
        for h in hypothesis_list:
            out.write(h+'\n')
    with open("./predicted_output/ground_truth.txt", mode='w', encoding='utf-8') as out:
        for r in reference_list:
            out.write(r+'\n')