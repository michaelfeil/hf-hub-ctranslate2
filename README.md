hf_hub_ctranslate2
==============================

Connecting Transformers on HuggingfaceHub with Ctranslate2 - a small utility for keeping tokenizer and model around Huggingface Hub.

[![codecov](https://codecov.io/gh/michaelfeil/hf-hub-ctranslate2/branch/main/graph/badge.svg?token=U9VIEFEELS)](https://codecov.io/gh/michaelfeil/hf-hub-ctranslate2)![CI pytest](https://github.com/michaelfeil/hf-hub-ctranslate2/actions/workflows/test_release.yml/badge.svg)

[Read the docs](https://michaelfeil.github.io/hf-hub-ctranslate2/)

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

--------
## Usage:

### PYPI Install
```bash
pip install hf-hub-ctranslate2
```
--------

## Decoder-only Transformer:
```python
# download ctranslate.Generator repos from Huggingface Hub (GPT-J, ..)
from hf_hub_ctranslate2 import TranslatorCT2fromHfHub, GeneratorCT2fromHfHub

model_name_1="michaelfeil/ct2fast-pythia-160m"
model = GeneratorCT2fromHfHub(
    # load in int8 on CPU
    model_name_or_path=model_name_1, device="cpu", compute_type="int8"
)
outputs = model.generate(
    text=["How do you call a fast Flan-ingo?", "User: How are you doing?"]
    # add arguments specifically to ctranslate2.Generator here
)
```
## Encoder-Decoder:
```python
from hf_hub_ctranslate2 import TranslatorCT2fromHfHub
# download ctranslate.Translator repos from Huggingface Hub (T5, ..)
model_name_2 = "michaelfeil/ct2fast-flan-alpaca-base"
model = TranslatorCT2fromHfHub(
        # load in int8 on CUDA
        model_name_or_path=model_name_2, device="cuda", compute_type="int8_float16"
)
outputs = model.generate(
    text=["How do you call a fast Flan-ingo?", "Translate to german: How are you doing?"],
    # use arguments specifically to ctranslate2.Translator below:
    min_decoding_length=8,
    max_decoding_length=16,
    max_input_length=512,
    beam_size=3
)
print(outputs)
```
## Encoder-Decoder for multilingual translations (m2m-100):
```python
from hf_hub_ctranslate2 import MultiLingualTranslatorCT2fromHfHub
model = MultiLingualTranslatorCT2fromHfHub(
    model_name_or_path="michaelfeil/ct2fast-m2m100_418M", device="cpu", compute_type="int8",
    tokenizer=AutoTokenizer.from_pretrained(f"facebook/m2m100_418M")
)

outputs = model.generate(
    ["How do you call a fast Flamingo?", "Wie geht es dir?"],
    src_lang=["en", "de"],
    tgt_lang=["de", "fr"]
)
```
## Encoder-only Sentence Transformers
Feel free to try out a new repo, using CTranslate2 for vector-embeddings: 
https://github.com/michaelfeil/infinity

```python
from hf_hub_ctranslate2 import CT2SentenceTransformer
model_name_pytorch = "intfloat/e5-small"
model = CT2SentenceTransformer(
    model_name_pytorch, compute_type="int8", device="cuda", 
)
embeddings = model.encode(
    ["I like soccer", "I like tennis", "The eiffel tower is in Paris"],
    batch_size=32,
    convert_to_numpy=True,
    normalize_embeddings=True,
)
print(embeddings.shape, embeddings)
scores = (embeddings @ embeddings.T) * 100
```

## Encoder-only -> no longer recommended
```python
from hf_hub_ctranslate2 import EncoderCT2fromHfHub
model_name = "michaelfeil/ct2fast-e5-small"
model = EncoderCT2fromHfHub(
        # load in int8 on CUDA
        model_name_or_path=model_name,
        device="cuda",
        compute_type="int8_float16",
)
outputs = model.generate(
    text=["I like soccer", "I like tennis", "The eiffel tower is in Paris"],
    max_length=64,
)
```




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/michaelfeil/hf-hub-ctranslate2.svg?style=for-the-badge
[contributors-url]: https://github.com/michaelfeil/hf-hub-ctranslate2/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/michaelfeil/hf-hub-ctranslate2.svg?style=for-the-badge
[forks-url]: https://github.com/michaelfeil/hf-hub-ctranslate2/network/members
[stars-shield]: https://img.shields.io/github/stars/michaelfeil/hf-hub-ctranslate2.svg?style=for-the-badge
[stars-url]: https://github.com/michaelfeil/hf-hub-ctranslate2/stargazers
[issues-shield]: https://img.shields.io/github/issues/michaelfeil/hf-hub-ctranslate2.svg?style=for-the-badge
[issues-url]: https://github.com/michaelfeil/hf-hub-ctranslate2/issues
[license-shield]: https://img.shields.io/github/license/michaelfeil/hf-hub-ctranslate2.svg?style=for-the-badge
[license-url]: https://github.com/michaelfeil/hf-hub-ctranslate2/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/michael-feil
