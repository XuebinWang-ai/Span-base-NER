#! /bin/bash

cwsfeat="bert"
data="ontonotes4"

# span-based ner
nohup python -u ../run.py predict\
    -d=0 \
    -f=../exp/$data.span.ner.v2.4.d \
    --feat=$cwsfeat \
    --fdata=../data/$data/test.char.bmes \
    --fpred=../predict_result/$data.pred.test \
    > ../log/pred/$data.test.$cwsfeat.pred.log 2>&1 &


data="ontonotes5"
# span-based ner
nohup python -u ../run.py predict\
    -d=0 \
    -f=../exp/$data.span.ner.v2 \
    --feat=$cwsfeat \
    --fdata=../data/$data/test.char.bmes \
    --fpred=../predict_result/$data.pred.test \
    > ../log/pred/$data.test.$cwsfeat.pred.log 2>&1 &

