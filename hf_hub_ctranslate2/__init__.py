# -*- coding: utf-8 -*-
"""Compatability between Huggingface and Ctranslate2."""
__all__ = ["__version__", "TranslatorCT2fromHfHub", "GeneratorCT2fromHfHub"]
from hf_hub_ctranslate2.translate import TranslatorCT2fromHfHub, GeneratorCT2fromHfHub

__version__ = "0.0.1"
