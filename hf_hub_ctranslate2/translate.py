import ctranslate2
from transformers import AutoTokenizer

from typing import Any, Literal, Union, List, Optional
import os

from hf_hub_ctranslate2._private.utils import download_model


class CTranslate2ModelfromHuggingfaceHub:
    def __init__(
        self,
        model_name_or_path: str,
        device="cuda",
        device_index=0,
        compute_type="int8_float16",
        tokenizer: Union[None, AutoTokenizer] = None,
        hub_kwargs: dict = {},
    ):
        """CTranslate2 compatibility class for Translator and Generator"""
        if os.path.isdir(model_name_or_path):
            model_path = model_name_or_path
        else:
            model_path = download_model(model_name_or_path, hub_kwargs=hub_kwargs)

        self.model = self.ctranslate_class(
            model_path,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
        )

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, fast=True)
        else:
            self.tokenizer = tokenizer

    def _forward(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    def generate(self, text: Union[str, List[str]], *forward_args, **forward_kwds: Any):
        orig_type = list
        if isinstance(text, str):
            orig_type = str
            text = [text]
        token_list = [
            self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(p)) for p in text
        ]
        texts_out = self._forward(token_list, *forward_args, **forward_kwds)

        if orig_type == str:
            return texts_out[0]
        else:
            return texts_out


class TranslatorCT2fromHfHub(CTranslate2ModelfromHuggingfaceHub):
    def __init__(
        self,
        model_name_or_path: str,
        device="cuda",
        device_index=0,
        compute_type="int8_float16",
        tokenizer: Union[AutoTokenizer, None] = None,
        hub_kwargs={},
    ):
        self.ctranslate_class = ctranslate2.Translator
        super().__init__(
            model_name_or_path,
            device,
            device_index,
            compute_type,
            tokenizer,
            hub_kwargs,
        )

    def _forward(self, *args, **kwds):
        tokens_out = self.model.translate_batch(*args, **kwds)
        return [
            self.tokenizer.decode(
                self.tokenizer.convert_tokens_to_ids(tokens_out[i].hypotheses[0])
            )
            for i in range(len(tokens_out))
        ]

    def generate(self, text: Union[str, List[str]], *forward_args, **forward_kwds: Any):
        """_summary_

        Args:
            text (Union[str, List[str]]): Input texts
            max_batch_size (int, optional): Batch size. Defaults to 0.
            batch_type (str, optional): _description_. Defaults to "examples".
            asynchronous (bool, optional): Only False supported. Defaults to False.
            beam_size (int, optional): _description_. Defaults to 2.
            patience (float, optional): _description_. Defaults to 1.
            num_hypotheses (int, optional): _description_. Defaults to 1.
            length_penalty (float, optional): _description_. Defaults to 1.
            coverage_penalty (float, optional): _description_. Defaults to 0.
            repetition_penalty (float, optional): _description_. Defaults to 1.
            no_repeat_ngram_size (int, optional): _description_. Defaults to 0.
            disable_unk (bool, optional): _description_. Defaults to False.
            suppress_sequences (Optional[List[List[str]]], optional): _description_. Defaults to None.
            end_token (Optional[Union[str, List[str], List[int]]], optional): _description_. Defaults to None.
            return_end_token (bool, optional): _description_. Defaults to False.
            prefix_bias_beta (float, optional): _description_. Defaults to 0.
            max_input_length (int, optional): _description_. Defaults to 1024.
            max_decoding_length (int, optional): _description_. Defaults to 256.
            min_decoding_length (int, optional): _description_. Defaults to 1.
            use_vmap (bool, optional): _description_. Defaults to False.
            return_scores (bool, optional): _description_. Defaults to False.
            return_attention (bool, optional): _description_. Defaults to False.
            return_alternatives (bool, optional): _description_. Defaults to False.
            min_alternative_expansion_prob (float, optional): _description_. Defaults to 0.
            sampling_topk (int, optional): _description_. Defaults to 1.
            sampling_temperature (float, optional): _description_. Defaults to 1.
            replace_unknowns (bool, optional): _description_. Defaults to False.
            callback (_type_, optional): _description_. Defaults to None.

        Returns:
            Union[str, List[str]]: text as output, if list, same len as input
        """
        return super().generate(text, *forward_args, **forward_kwds)


class GeneratorCT2fromHfHub(CTranslate2ModelfromHuggingfaceHub):
    def __init__(
        self,
        model_name_or_path: str,
        device="cuda",
        device_index=0,
        compute_type: Literal["int8_float16", "int8"] = "int8_float16",
        tokenizer: Union[AutoTokenizer, None] = None,
        hub_kwargs={},
    ):
        self.ctranslate_class = ctranslate2.Generator
        super().__init__(
            model_name_or_path,
            device,
            device_index,
            compute_type,
            tokenizer,
            hub_kwargs,
        )

    def _forward(self, *args, **kwds):
        tokens_out = self.model.generate_batch(*args, **kwds)
        return [
            self.tokenizer.decode(tokens_out[i].sequences_ids[0])
            for i in range(len(tokens_out))
        ]

    def generate(self, text: Union[str, List[str]], *forward_args, **forward_kwds: Any):
        """_summary_

        Args:
            text (str | List[str]): Input texts
            max_batch_size (int, optional): _description_. Defaults to 0.
            batch_type (str, optional): _description_. Defaults to 'examples'.
            asynchronous (bool, optional): _description_. Defaults to False.
            beam_size (int, optional): _description_. Defaults to 1.
            patience (float, optional): _description_. Defaults to 1.
            num_hypotheses (int, optional): _description_. Defaults to 1.
            length_penalty (float, optional): _description_. Defaults to 1.
            repetition_penalty (float, optional): _description_. Defaults to 1.
            no_repeat_ngram_size (int, optional): _description_. Defaults to 0.
            disable_unk (bool, optional): _description_. Defaults to False.
            suppress_sequences (Optional[List[List[str]]], optional): _description_. Defaults to None.
            end_token (Optional[Union[str, List[str], List[int]]], optional): _description_. Defaults to None.
            return_end_token (bool, optional): _description_. Defaults to False.
            max_length (int, optional): _description_. Defaults to 512.
            min_length (int, optional): _description_. Defaults to 0.
            include_prompt_in_result (bool, optional): _description_. Defaults to True.
            return_scores (bool, optional): _description_. Defaults to False.
            return_alternatives (bool, optional): _description_. Defaults to False.
            min_alternative_expansion_prob (float, optional): _description_. Defaults to 0.
            sampling_topk (int, optional): _description_. Defaults to 1.
            sampling_temperature (float, optional): _description_. Defaults to 1.

        Returns:
            str | List[str]: text as output, if list, same len as input
        """
        return super().generate(text, *forward_args, **forward_kwds)
