import pytorch_lightning as pl
import argparse
import os
import pickle
import naturalproofs.dataloaders as dataloaders
import naturalproofs.model as mutils
import torch
import transformers
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import naturalproofs.model_bert_no_training as model_bert
import json

# use default bert (bert-base-cased)
tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-cased')

# load model
model = mutils.Classifier.load_from_checkpoint('ckpt\pairwise_proofwiki.ckpt')

print("Loading data")
ds_raw = pickle.load(open('data\pairwise_proofwiki__bert-base-cased.pkl', 'rb'))

xdl, rdl, x2rs = dataloaders.get_eval_dataloaders(
        ds_raw,
        pad=tokenizer.pad_token_id,
        token_limit=8192,
        workers=0,
        split_name='test'
    )
print("%d examples\t%d refs" % (len(xdl.dataset.data), len(rdl.dataset.data)))

ref_dl = rdl
ex_dl = xdl

model.eval()
model.cuda()
torch.set_grad_enabled(False)
r_encs, rids = model.pre_encode_refs(ref_dl, progressbar=True)

# prepare data (see from analyze.py)
print("Loading data")
raw_ds = json.load(open('.\\data\\naturalproofs_proofwiki.json', 'r'))
global refs
refs = raw_ds['dataset']['theorems'] + raw_ds['dataset']['definitions'] + raw_ds['dataset']['others']
global id2ref
id2ref = {ref['id'] : ref for ref in refs}


from flask import Flask, request, send_from_directory
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    inputs = data['raw']

    ids = tokenizer.encode(inputs)
    x = torch.tensor([ids])
    x = x.cuda()
    x_enc = model.encode_x(x) # (B, D)
    # given x theorem, and many references, get the reference rank
    logits = model.forward_clf(x_enc, r_encs) # (B, R)

    predicts = []
    for b in range(logits.size(0)):
        ranked = list(zip(logits[b].tolist(), rids.tolist()))
        ranked = sorted(ranked, reverse=True)

        for top5 in range(5):
            related_rid = ranked[top5][1]
            predicts.append(id2ref[related_rid]['title'])

    return {
        "thms": predicts
    }


@app.route('/')
def index():
    return send_from_directory('.', 'demo.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)