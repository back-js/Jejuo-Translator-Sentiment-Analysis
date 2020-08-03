# 제주어 <-> 표준어 학습

- 학습은 Colab pro로 진행하였습니다.
- 먼저 라이브러리를 설치합니다.
``` python
!pip install sentencepiece
!pip install jamo
!pip install fairseq
```

``` python
%%bash
python '/content/drive/My Drive/jejueo/translation/bpe_segment.py' --jit '/content/drive/My Drive/jejueo/JIT_dataset/' --vocab_size 4000
```

``` python
%%bash
#### 단어 사전 만들기
python '/content/drive/My Drive/jejueo/translation/prepro.py' --src ko --tgt je --vocab_size 4000
python '/content/drive/My Drive/jejueo/translation/prepro.py' --src je --tgt ko --vocab_size 4000
```

``` python
%%bash
#### 표준어 -> 제주어 train
export lang1="ko"
export lang2="je"
fairseq-train data/4k/${lang1}-${lang2}-bin \
    --arch transformer       \
    --optimizer adam \
    --lr 0.0005 \
    --label-smoothing 0.1 \
    --dropout 0.3       \
    --max-tokens 4000 \
    --min-lr '1e-09' \
    --lr-scheduler inverse_sqrt       \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy       \
    --max-epoch 100 \
    --warmup-updates 4000 \
    --warmup-init-lr '1e-07'    \
    --adam-betas '(0.9, 0.98)'       \
    --save-dir train/4k/${lang1}-${lang2}/ckpt  \
    --save-interval 10
```

``` python
%%bash
#### 제주어 -> 표준어 train
export lang1="je"
export lang2="ko"
fairseq-train data/4k/${lang1}-${lang2}-bin \
    --arch transformer       \
    --optimizer adam \
    --lr 0.0005 \
    --label-smoothing 0.1 \
    --dropout 0.3       \
    --max-tokens 4000 \
    --min-lr '1e-09' \
    --lr-scheduler inverse_sqrt       \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy       \
    --max-epoch 80 \
    --warmup-updates 4000 \
    --warmup-init-lr '1e-07'    \
    --adam-betas '(0.9, 0.98)'       \
    --save-dir train/4k/${lang1}-${lang2}/ckpt  \
    --save-interval 10
```

``` python
%%bash
#### 학습데이터를 바탕으로 BLEU값 평가

export lang1="ko"
export lang2="je"

fairseq-generate data/4k/${lang1}-${lang2}-bin \
  --path '/content/train/4k/je-ko/ckpt/checkpoint10.pt'
```