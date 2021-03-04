---
math: true
---

# 논문 REVIEW - Neural Machine Translation by Jointly Learning to Align and Translate

## Hong Gibong / 2021년 3월 4일

* 해당 논문의 원본은 https://arxiv.org/abs/1409.0473 에서 확인할 수 있습니다.
* 현재 작성된 버전은 미완성본입니다. 스터디 발표 이후에 부족한 내용을 보완해서 최종 업로드할 예정입니다. 참고 부탁드립니다.
* 특히 논문 내용 중 수식과 관련한 내용은 수업 중에 보다 더 자세히 설명드리겠습니다 :)

해당 논문이 세상에 나온 지 5년이 지났습니다. 그리고 현재 attention은 여러 분야에서 단일 기술로 활용되고 있습니다. 기계 번역의 퀄리티를 높이기 위해 RNN/LSTM에 대한 보조적 수단으로 사용된 attention이 self-attention으로 발전했고, 오직 이 attention 메커니즘 만으로 Transformer를 구성했습니다. Transformer는 최근 자연어생성(NLG) 및 자연어이해(NLU) 관련 다양한 task에서 SOTA를 기록하는데 기여했습니다. 자연어처리 분야의 일부 연구자들은 Attention의 등장을 '단어 임베딩'만큼이나 중요한 turning point로 생각하는데, 성능을 올려주었을 뿐만 아니라 Alignment Matrix를 통해 그 해석 가능성을 만들어주었다는 점에서 충분히 높은 평가를 받을 만하다고 저 역시도 생각합니다.(Alignment Matrix에 대해서는 논문 본문에서 설명하겠습니다.)

![seq2seq_bottleneck](./images/posts/210304/seq2seq_bottleneck.png)

이미지: Seq2seq 모델의 bottleneck problem 설명
출처: cs224n-2021-lecture07-nmt 강의안

## 1. Introduction

Attention은 기존 RNN 기반 encoder-decoder 모델의(Seq2Seq) 단일 context vector의 한계를 개선하기 위해 등장했습니다. 기존 Seq2seq 모델에서는 encoder에서 출력된 마지막 hidden state 정보를 단일 context vector로 사용했습니다. 물론 이 구조만으로도 기계 번역의 성능은 향상됐습니다. 하지만 input sequence의 길이가 길어질 수록 하나의 context vector가 해당 input의 모든 정보를 담아내지 못하는 문제가 발생합니다. 이를 흔히 병목현상(bottleneck problem)이라고 합니다. 실제로 해당 논문의 저자이자, Seq2Seq 모델을 처음 발표한 조경현 교수가 2014년 발표한 논문에서 문장의 길이가 길어질 수록 기존의 encoder-decoder 구조의 성능이 급격하게 저하됨을 보였습니다.

해당 논문에서 제시하는 모델의 키포인트는 다음과 같습니다.

> It does not attempt to encode a whole input sentence into a single fixed-length vector. 
>
> Instead, it encodes the input sentence into a sequence of vectors and chooses a subset of these vectors adaptively while decoding the translation.

즉 기계 번역 모델이 더 이상 source sequence의 모든 정보를 제약된 공간에 짜넣듯이 '압축'할 필요가 없어졌다는 것입니다. 특히 번역 작업의 경우 대상 시퀀스가 길어질 수록 참조해야 하는 source sequence의 정보가 제각기 다를텐데, 기존의 모델은 모두 같은 context vector를 공유하면서 hidden state를 만들어낸다는 점에서 근원적인 한계를 지녔던 것입니다.



## 2. Background: Neural Machine Translation

인공신경망 기반의 기계 번역은 대개 'source sentence **x**'를 인코딩해서 'target sentence **y**'로 디코딩하는 두 가지 단계로 진행됩니다. 이러한 RNN 기반의 Encoder-Decoder에 대해 간단히 표현하자면, source sentence가 주어졌을 때 대상이 되는 target sentence가 등장할 조건부 확률을 최대화하는 번역 작업을 진행하는 것입니다. 이를 수식으로 나타내면 다음과 같습니다.
$$
p(y) = \prod_{t=1}^{T} p(y_t | y_1,...,y_{t-1}, c)
$$
여기서 사용된 $c$ 는 디코더의 모든 time step에서 공유하는 context vector로, 이는 encoder의 매 time step에서 매핑되는 hidden states의 비선형결합에 의해 정의됩니다. 이를 수식으로 표현하면 다음과 같습니다.
$$
c = q(h_1,...,h_{T_x})
$$
Encoder에서 출력되는 각 hidden states는 이전 타임스텝에서의 hidden state와 input word(token) embedding의 비선형결합으로 생성됩니다. 이 역시 수식으로 표현하면 다음과 같습니다. 참고로 Sutskever et al.(2014)의 seq2seq 논문에서는 비선형결합(수식의 $f$,$q$)으로 LSTM을 사용합니다.
$$
h_t = f(x_t, h_{t-1})
$$


## 3. Learning to ALIGN and TRANSLATE

본 논문에서 제시하는 모델 역시 그 기본은 이전의 RNN Encoder-Decoder에 기초하고 있습니다. 하지만 그 설계 방식에는 encoder와 decoder 부분에 모두 차이가 있습니다. 아래 그림을 보면서 설명드리겠습니다.

![model_arch](images/posts/210304/model_arch.png)

### (1) Encoder

Encoder 부분은 상대적으로 기존 seq2seq 모델의 encoder와 유사합니다. 다만 기존 모델이 정방향의(forward) RNN을 사용해서 input source를 mapping했다면, 논문에서 제시하는 모델은 bidirectional RNN을 사용했다는 점에 차이가 있습니다. 쉽게 말해 정방향과 역방향의 RNN을 모두 사용해서   source sentence에 대한 hidden states를 각각 생성하고 나서, 두 개의 hidden states를 time step 별로 concat하는 것입니다. 이렇게 concat한 벡터를 해당 논문에서는 'annotation'이라고 따로 이름을 붙였습니다. RNN을 두 번 반복하는 것은 모델이 단어를 더 잘 기억하는데 도움을 줍니다. 또한 RNN 특성상 근처 단어의 정보를 더 많이 표상하기 때문에, $i$번째 time step의 annotation은 $i$번째 input word 주변 단어들의 정보를 더 많이 담고 있습니다.

### (2) Decoder

해당 논문에서 제시하는 모델의 가장 큰 특징, 즉 attention을 활용하는 방식이 이 부분에서 나타납니다. 앞서 얘기한 것처럼 단일 context vector를 공유해서 decoding을 진행할 경우 1) 고정된 길이의 vector를 사용한다는 점, 2) decoding 과정에서 input source에 대한 focus를 다양하게 가져갈 수 없다는 점에서 한계를 지닙니다.

새로운 모델의 decoder에서 각각의 time step에 따른 조건부 확률은 다음과 같이 정의됩니다.

*(설명에 앞서, 해당 파트의 모든 수식에서 $i$는 decoder의 time step, $j$는 encoder의 time step임을 알려드립니다. 헷갈리지 마세요!)*
$$
p(y_i|y_1,...,y_{i-1}, x) = g(y_{i-1}, s_i, c_i)
$$

$$
s_i = f(s_{i-1}, y_{i-1}, c_i)
$$

기존의 seq2seq 모델과 달리, 새로운 모델의 decoder에서는 $c_i$를 context vector로 활용합니다. 즉 encoder 부분에서 생성된 annotations를 바탕으로 $c_i$가 결정되는 것입니다. $c_i$에 대한 수식을 적어보면 다음과 같습니다.
$$
c_i = \sum_{j=1}^{T_x} \alpha_{ij}h_j
$$
여기서 $a_{ij}$는 가중치를 의미하며, neural network에 의해 학습됩니다. $h_j$는 encoder 부분에서의 j번째 input의 annotation입니다. 즉 input sequence에서 각 단어에 대해 어느 정도의 가중치를 부여할 것인지를 정하고 이를 다 합한 것이 $c_i$ 입니다.

그렇다면 다시 $a_{ij}$, 즉 'input의 j번째 단어와 output의 i번째 단어가 얼마나 관련이 있는지를 의미하는 가중치'를 구하는 수식을 알아보겠습니다.

![스크린샷 2021-03-04 오전 1.49.31](/Users/gibonghong/Library/Application Support/typora-user-images/스크린샷 2021-03-04 오전 1.49.31.png
$$
\alpha_{ij} = exp(e_{ij})/\sum_{j=1}^{T_x}exp(e_{ik})
$$

$$
e_{ij} = a(s_{i-1}, h_j)
$$

$a_{ij}$는 $e_{ij}$를 확률값으로 변환한 것입니다(softmax). 그리고 $e_{ij}$는 decoder의 time step $i-1$의 hidden state와 encoder의 annotation 사이 alignment model이 적용된 결과입니다. 여기 alignment model a는 feed-forward neural network로 정의됩니다. Weight는 $s_{i-1}$과 $h_j$를 input으로 한 feed-forward neural network에 의해 형성됩니다.

해당 논문에서는 RNN 계열의 activation function $f$와 alignment model $a$를 일반화해서 정의하는데, 실제로 논문에 제시된 모델에서는 $f$로 GRU(Cho et al., 2014)를 사용했고, alignment model로는 bahdanau attention을 사용했습니다. 이는 우리가 일반적으로 알고 있는 dot-product attention과 조금 다른 개념입니다. 우리는 위 수식의 $s_{i-1}$과 $h_j$, 즉 decoder의 특정 hidden state와 encoder의 특정 annotation 사이 'dot product'를 한 결과를 바탕으로 attention 개념을 처음 이해했습니다.

결과적으로 가중치 $a_{ij}$는 'decoder 기준 바로 이전 time step인 $s_{i-1}$'에 대한 encoder의 time step j의 annotation $h_j$의 관련성을 반영하는 것입니다. 이 관련성을 바탕으로 decoder는 time step $i$에서의 hidden state를 결정하게 되고 다음 단어인 $y_i$를 생성하는 것입니다.



## 4. Experiment Settings / Results

![result_sentence_len](images/posts/210304/result_sentence_len.png)

해당 실험의 결과는 놀라웠습니다. 실험에서는 총 2가지 타입의 모델을 사용했습니다. 첫번째 모델로는 논문에서 제시한 RNNsearch, 두번째 모델로는 기존 모델인 RNNenc입니다. 실험을 위해서 저자들은 각 모델을 2차례 학습시켰는데, 처음에는 max length를 30 words로 설정했고 그 다음에는 50 words로 설정했습니다. 그리하여 총 4가지 모델에 대한 실험을 진행했고 위의 그림과 같은 결과를 보였습니다.

RNNsearch는 문장의 길이가 30 이상일 때 BLEU score 기준으로 높은 성능을 유지하면서 다른 모델들과 큰 격차를 보였습니다. 뿐만 아니라 문장의 길이가 짧을 때에도 RNNenc보다 높은 성능을 보였습니다. 즉 기존 모델이 갖고 있던 fixed-length 단일 벡터가 지닌 한계를 이번 실험을 통해 증명한 것입니다.

Alignment의 결과는 아래 그림과 같았습니다. 해당 그림은 논문에서 영어 source sentence - 프랑스어 generated sentence pair를 각각 x축과 y축에 나타낸 것입니다. 두 단어 사이 연관성이 0에 가까울 수록 검정색, 1에 가까울 수록 하얀색입니다. 프랑스어를 잘 알지 못한다 하더라도 accord-agreement, economique-Economic 등 같은 의미의 단어가 높은 연관성을 지닐 수 있음을 확인할 수 있습니다. 

영어-프랑스어 문장 사이 연관성이 대체로 monotonic한 관계성을 지니지만 한편으로는 non-monotonic한 관계를 관찰할 수 있습니다. 실제로 그림 (a)를 보면 모델이 [European Economic Area]를 [zone economique europeen]으로 번역했음을 알 수 있습니다. 또한 해당 그림을 통해 모델이 'soft-alignment', 즉 하나의 단어가 두 개 이상의 단어 정보를 조합하여 생성되거나(many-to-one) 혹은 그 반대의 경우(one-to-many)도 존재할 수 있음을 알 수 있습니다. 이는 'hard-alignment'에 비해 확실히 더 유연한 번역을 가능하게 합니다.

![result_alignment_matrix](images/posts/210304/result_alignment_matrix.png)

특히 해당 그림은 기존에 해석이 어려웠던 neural network 기반의 번역 모델을 시각화했다는 점에서 큰 의의를 지닙니다. 모델이 source input과 generated output 사이에서 어떠한 방식으로 mapping을 진행하고 있는지 확인할 수 있기 때문입니다.

## 5. Conclusion

해당 논문은 모델이 soft-search를 통해 encoder에 의해 생성된 annotations를 활용함으로써, 기존의 fixed-length vector에서 벗어나 긴 문장에서도 잘 작동하는 번역 성능을 달성했다는 점에서 큰 의의를 지닙니다. 무엇보다 기존의 phrase-based statistical 기계 번역에 필적할 만한 성능을 기록했다는 점에서 짧은 기간 안에 엄청난 성장을 이뤘다고 볼 수 있습니다.