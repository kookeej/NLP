🤗 BERT 🤗
===
> *https://huggingface.co/docs/transformers/model_doc/bert를 참고하여 작성하였습니다.*
# Contents
> * **[BertConfig](#BertConfig)**
> * **[BertModel](#BertModel)**

***

## BertConfig
* Configuration class
* `BertModel`과 `TFBertModel`의 구성을 저장
* 지정된 arguments에 따라 BERT를 인스턴스화
* default configuration으로 인스턴스화하면 `bert-base-uncased` 아키텍쳐와 비슷한 configuration을 산출한다.
* Configuration objects는 `PretrainedConfig`로부터 상속된 객체이고, **모델의 출력값을 제어**하는데 사용된다.

```python
from transformers import BertModel, BertConfig

# bert-base-uncased style configuration 초기화
configuration = BertConfig()

# bert-base-uncased style configuration으로부터 모델 초기화
model = BertModel(configuration)

# 모델 configuration에 접근
configuration = model.config
```

## BertModel
```
(config, add_pooling_layer=True)
```
**config**(BertConfig)
> * 모델의 모든 파라미터를 가진 Model configuration class.
> * config file로 초기화하면 모델과 관련된 weight들은 로드되지 않고 configuration만 로드된다.
> * `from_pretrained()` 메소드를 사용하여 모델 weights를 로드한다.

* 맨 위에 특정 head가 없는 raw hidden-states를 출력하는 bare Bert Model transformer.
* `PreTrainedModel`상속 받음.
* `torch.nn.Module` subclass
* Decoder는 물론이고 self-attention만을 가진 Encoder의 역할을 한다.
* Decoder의 역할을 하기 위해서는 `is_decoder`가 `True`로 초기화 되어야 한다.
* `Seq2Seq`모델로 사용하려면 `is_decoder`와 `add_cross_attention`이 `True`로 설정되어야 함.
  * `encoder_hidden_states`는 forward pass의 input으로 들어갈 수 있다.


```python
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)
# **inputs: keyword argument만 받을 때
# (키워드=특정값) 형태로 함수를 호출할 수 있음
# {'키워드': '특정 값'} 딕셔너리 형태로 함수 내부로 전달된다.

last_hidden_states = outputs.last_hidden_state
```
