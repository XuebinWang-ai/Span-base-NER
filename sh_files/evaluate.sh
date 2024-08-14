#! /bin/bash

cwsfeat="bert"
data="ontonotes4"

# span-based ner
nohup python -u ../run.py evaluate \
    -d=3 \
    -f=../exp/$data.span.ner.v2.4.d \
    --feat=$cwsfeat \
    --fdata=../data/$data/test.char.bmes \
    --fprob=../predict_result/$data.test.prob.txt \
    --max_ne_length=-1 \
    --threshold=0.1 \
    > ../log/pred/$data.test.$cwsfeat.eval.log 2>&1 &


data="ontonotes5"
# span-based ner
nohup python -u ../run.py evaluate \
    -d=4 \
    -f=../exp/$data.span.ner.v2 \
    --feat=$cwsfeat \
    --fdata=../data/$data/test.char.bmes \
    --fprob=../predict_result/$data.test.prob.txt \
    --max_ne_length=-1 \
    --threshold=0.1 \
    > ../log/pred/$data.test.$cwsfeat.eval.log 2>&1 &

