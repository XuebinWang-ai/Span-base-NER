# Span-based NER (Semi-Markov)

## Training 
* Train a BERT-MLP-Biaffine model.
* BERT model can be "bert-base-chinese" or "bert-wwm"
* loss = logZ - gold_scores

```shell
data="ontonotes4"   # or data="ontonotes5"
nohup python -u ../run.py train \
    -p \
    -d=0 \
    --feat=$cwsfeat \
    --ftrain=../data/$data/train.char.bmes \
    --fdev=../data/$data/dev.char.bmes \
    --ftest=../data/$data/test.char.bmes \
    --max_ne_length=-1 \
    -f=../exp/$data.span.ner.wwm \
    > ../log/$data.span.ner.wwm.log 2>&1 &
```

-d: use which gpu

--max_ne_length: the maximum NE length to search. "-1" means do not use this parameter.
In the ontonotes4 and ontonotes5, the maximum NE lengths are "23" and "55" respectively.

--cwsfeat: bert

## Evaluate
* We can get the probability of NE predicted by the model.
* The probability equals to "LogZ" derivative with respect to the model output "scores".

```shell
nohup python -u ../run.py evaluate \
    -d=0 \
    -f=../exp/$data.span.ner \
    --feat=$cwsfeat \
    --fdata=../data/$data/test.char.bmes \
    --fprob=../predict_result/$data.test.prob.txt \
    --max_ne_length=-1 \
    --threshold=0.1 \
    > ../log/pred/$data.test.$cwsfeat.eval.log 2>&1 &
```

--threshold: the minmum probability of NE to save.

--fprob: the path to save 1-best results and all NE with probability larger than or equal to threshold.

## Predict
```shell
nohup python -u ../run.py predict\
    -d=0 \
    -f=../exp/$data.span.ner \
    --feat=$cwsfeat \
    --fdata=../data/$data/test.char.bmes \
    --fpred=../predict_result/$data.pred.test \
    --max_ne_length=-1 \
    > ../log/pred/$data.test.$cwsfeat.pred.log 2>&1 &
```

--fpred: the path to save the predict results in BMES form.

