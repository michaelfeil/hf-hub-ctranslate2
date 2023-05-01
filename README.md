hf_hub_ctranslate2
==============================

Connecting Transfromers on HuggingfaceHub with Ctranslate2 - a small utility for keeping tokenizer and model around Huggingface Hub.

[![codecov](https://codecov.io/gh/michaelfeil/hf-hub-ctranslate2/branch/master/graph/badge.svg?token=U9VIEFEELS)](https://codecov.io/gh/michaelfeil/hf-hub-ctranslate2)![CI pytest](https://github.com/michaelfeil/hf-hub-ctranslate2/actions/workflows/test_release.yml/badge.svg)

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
)

# download ctranslate.Translator repos from Huggingface Hub (T5, ..)
model_name_2 = "michaelfeil/ct2fast-flan-alpaca-base"
model = GeneratorCT2fromHfHub(
        # load in int8 on CUDA
        model_name_or_path=model_name_2, device="cuda", compute_type="int8_float16"
)
outputs = model.generate(
    text=["How do you call a fast Flan-ingo?", "Translate to german: How are you doing?"],
    min_decoding_length=24,
    max_decoding_length=32,
    max_input_length=512,
    beam_size=5
)
print(outputs)
```

--------
## PYPI Install
```bash
pip install hf-hub-ctranslate2
```
--------



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
