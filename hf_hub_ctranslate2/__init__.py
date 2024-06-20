# -*- coding: utf-8 -*-
"""Compatability between Huggingface and Ctranslate2."""
# __all__ = ["__version__", "TranslatorCT2fromHfHub", "GeneratorCT2fromHfHub", "MultiLingualTranslatorCT2fromHfHub"]
import importlib.metadata
import os
from hf_hub_ctranslate2.translate import (
    TranslatorCT2fromHfHub,
    GeneratorCT2fromHfHub,
    MultiLingualTranslatorCT2fromHfHub,
    EncoderCT2fromHfHub,
)
from hf_hub_ctranslate2.ct2_sentence_transformers import CT2SentenceTransformer

__version__ = importlib.metadata.version("hf_hub_ctranslate2")
