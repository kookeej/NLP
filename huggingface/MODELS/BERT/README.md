🤗 BERT 🤗
===
reference
> https://huggingface.co/docs/transformers/model_doc/bert
# Contents
> * **[BertConfig](#BertConfig)**

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
