# -*- coding: utf-8 -*-
"""Compatability between Huggingface and Ctranslate2."""
# __all__ = ["__version__", "TranslatorCT2fromHfHub", "GeneratorCT2fromHfHub", "MultiLingualTranslatorCT2fromHfHub", "_private"]
from hf_hub_ctranslate2.translate import TranslatorCT2fromHfHub, GeneratorCT2fromHfHub, MultiLingualTranslatorCT2fromHfHub
from hf_hub_ctranslate2.util import _download_model
__version__ = "2.0.6"