#! /bin/bash

cwsfeat="bert"
data="ontonotes4"
# data="ontonotes5"

nohup python -u ../run.py train \
    -p \
    -d=4 \
    --feat=$cwsfeat \
    --ftrain=../data/$data/train.char.bmes \
    --fdev=../data/$data/dev.char.bmes \
    --ftest=../data/$data/test.char.bmes \
    --max_ne_length=-1 \
    -f=../exp/$data.span.ner.wwm \
    > ../log/$data.span.ner.wwm.log 2>&1 &

data="ontonotes5"
nohup python -u ../run.py train \
    -p \
    -d=3 \
    --feat=$cwsfeat \
    --ftrain=../data/$data/train.char.bmes \
    --fdev=../data/$data/dev.char.bmes \
    --ftest=../data/$data/test.char.bmes \
    --max_ne_length=-1 \
    -f=../exp/$data.span.ner.wwm \
    > ../log/$data.span.ner.wwm.log 2>&1 &

