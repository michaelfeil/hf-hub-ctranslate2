import ctranslate2
import functools

try:
    from transformers import AutoTokenizer

    autotokenizer_ok = True
except ImportError:
    AutoTokenizer = object
    autotokenizer_ok = False

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from typing import Any, Union, List
import os
import glob
import shutil
from hf_hub_ctranslate2.util import utils as _utils


class CTranslate2ModelfromHuggingfaceHub:
    """CTranslate2 compatibility class for Translator and Generator"""

    def __init__(
        self,
        model_name_or_path: str,
        device: Literal["cpu", "cuda"] = "cuda",
        device_index=0,
        compute_type: Literal["int8_float16", "int8"] = "int8_float16",
        tokenizer: Union[AutoTokenizer, None] = None,
        hub_kwargs: dict = {},
        **kwargs: Any,
    ):
        # adaptions from https://github.com/guillaumekln/faster-whisper
        if os.path.isdir(model_name_or_path):
            model_dir = model_name_or_path
        else:
            try:
                model_dir = _utils._download_model(
                    model_name_or_path, hub_kwargs=hub_kwargs
                )
            except Exception:
                hub_kwargs["local_files_only"] = True
                model_dir = _utils._download_model(
                    model_name_or_path, hub_kwargs=hub_kwargs
                )

        model_bin = os.path.join(model_dir, "model.bin")
        if not os.path.exists(model_bin):
            #
            shards = glob.glob(model_bin.replace(".bin", "-.*of.*bin"))
            shards = sorted(shards, key=lambda path: int(path.split(".")[-1]))
            with open(model_bin, "wb") as model_bin_file:
                for shard in shards:
                    with open(shard, "rb") as shard_file:
                        shutil.copyfileobj(shard_file, model_bin_file)
                    os.remove(shard)
        self.model = self.ctranslate_class(
            model_dir,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            **kwargs,
        )

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            if "tokenizer.json" in os.listdir(model_dir):
                if not autotokenizer_ok:
                    raise ValueError(
                        "`pip install transformers` missing to load AutoTokenizer."
                    )
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir, fast=True)
            else:
                raise ValueError(
                    "no suitable Tokenizer found. "
                    "Please set one via tokenizer=AutoTokenizer.from_pretrained(..) arg."
                )

    def _forward(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    def tokenize_encode(self, text, *args, **kwargs):
        return [
            self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(p)) for p in text
        ]

    def tokenize_decode(self, tokens_out, *args, **kwargs):
        raise NotImplementedError

    def generate(
        self,
        text: Union[str, List[str]],
        encode_kwargs={},
        decode_kwargs={},
        *forward_args,
        **forward_kwds: Any,
    ):
        orig_type = list
        if isinstance(text, str):
            orig_type = str
            text = [text]
        token_list = self.tokenize_encode(text, **encode_kwargs)
        tokens_out = self._forward(token_list, *forward_args, **forward_kwds)
        texts_out = self.tokenize_decode(tokens_out, **decode_kwargs)
        if orig_type == str:
            return texts_out[0]
        else:
            return texts_out


class TranslatorCT2fromHfHub(CTranslate2ModelfromHuggingfaceHub):
    def __init__(
        self,
        model_name_or_path: str,
        device: Literal["cpu", "cuda"] = "cuda",
        device_index=0,
        compute_type: Literal["int8_float16", "int8"] = "int8_float16",
        tokenizer: Union[AutoTokenizer, None] = None,
        hub_kwargs={},
        **kwargs: Any,
    ):
        """for ctranslate2.Translator models, in particular m2m-100

        Args:
            model_name_or_path (str): _description_
            device (Literal[cpu, cuda], optional): _description_. Defaults to "cuda".
            device_index (int, optional): _description_. Defaults to 0.
            compute_type (Literal[int8_float16, int8], optional): _description_. Defaults to "int8_float16".
            tokenizer (Union[AutoTokenizer, None], optional): _description_. Defaults to None.
            hub_kwargs (dict, optional): _description_. Defaults to {}.
            **kwargs (Any, optional): Any additional arguments
        """
        self.ctranslate_class = ctranslate2.Translator
        super().__init__(
            model_name_or_path,
            device,
            device_index,
            compute_type,
            tokenizer,
            hub_kwargs,
            **kwargs,
        )

    def _forward(self, *args, **kwds):
        return self.model.translate_batch(*args, **kwds)

    def tokenize_decode(self, tokens_out, *args, **kwargs):
        return [
            self.tokenizer.decode(
                self.tokenizer.convert_tokens_to_ids(tokens_out[i].hypotheses[0]),
                *args,
                **kwargs,
            )
            for i in range(len(tokens_out))
        ]

    def generate(
        self,
        text: Union[str, List[str]],
        encode_tok_kwargs={},
        decode_tok_kwargs={},
        *forward_args,
        **forward_kwds: Any,
    ):
        """_summary_

        Args:
            text (Union[str, List[str]]): Input texts
            encode_tok_kwargs (dict, optional): additional kwargs for tokenizer
            decode_tok_kwargs (dict, optional): additional kwargs for tokenizer
            max_batch_size (int, optional): Batch size. Defaults to 0.
            batch_type (str, optional): _. Defaults to "examples".
            asynchronous (bool, optional): Only False supported. Defaults to False.
            beam_size (int, optional): _. Defaults to 2.
            patience (float, optional): _. Defaults to 1.
            num_hypotheses (int, optional): _. Defaults to 1.
            length_penalty (float, optional): _. Defaults to 1.
            coverage_penalty (float, optional): _. Defaults to 0.
            repetition_penalty (float, optional): _. Defaults to 1.
            no_repeat_ngram_size (int, optional): _. Defaults to 0.
            disable_unk (bool, optional): _. Defaults to False.
            suppress_sequences (Optional[List[List[str]]], optional): _.
               Defaults to None.
            end_token (Optional[Union[str, List[str], List[int]]], optional): _.
               Defaults to None.
            return_end_token (bool, optional): _. Defaults to False.
            prefix_bias_beta (float, optional): _. Defaults to 0.
            max_input_length (int, optional): _. Defaults to 1024.
            max_decoding_length (int, optional): _. Defaults to 256.
            min_decoding_length (int, optional): _. Defaults to 1.
            use_vmap (bool, optional): _. Defaults to False.
            return_scores (bool, optional): _. Defaults to False.
            return_attention (bool, optional): _. Defaults to False.
            return_alternatives (bool, optional): _. Defaults to False.
            min_alternative_expansion_prob (float, optional): _. Defaults to 0.
            sampling_topk (int, optional): _. Defaults to 1.
            sampling_temperature (float, optional): _. Defaults to 1.
            replace_unknowns (bool, optional): _. Defaults to False.
            callback (_type_, optional): _. Defaults to None.

        Returns:
            Union[str, List[str]]: text as output, if list, same len as input
        """
        return super().generate(
            text,
            encode_kwargs=encode_tok_kwargs,
            decode_kwargs=decode_tok_kwargs,
            *forward_args,
            **forward_kwds,
        )


class MultiLingualTranslatorCT2fromHfHub(CTranslate2ModelfromHuggingfaceHub):
    def __init__(
        self,
        model_name_or_path: str,
        device: Literal["cpu", "cuda"] = "cuda",
        device_index=0,
        compute_type: Literal["int8_float16", "int8"] = "int8_float16",
        tokenizer: Union[AutoTokenizer, None] = None,
        hub_kwargs={},
        **kwargs: Any,
    ):
        """for ctranslate2.Translator models

        Args:
            model_name_or_path (str): _description_
            device (Literal[cpu, cuda], optional): _description_. Defaults to "cuda".
            device_index (int, optional): _description_. Defaults to 0.
            compute_type (Literal[int8_float16, int8], optional): _description_. Defaults to "int8_float16".
            tokenizer (Union[AutoTokenizer, None], optional): _description_. Defaults to None.
            hub_kwargs (dict, optional): _description_. Defaults to {}.
            **kwargs (Any, optional): Any additional arguments
        """
        self.ctranslate_class = ctranslate2.Translator
        super().__init__(
            model_name_or_path,
            device,
            device_index,
            compute_type,
            tokenizer,
            hub_kwargs,
            **kwargs,
        )

    def _forward(self, *args, **kwds):
        target_prefix = [
            [self.tokenizer.lang_code_to_token[lng]] for lng in kwds.pop("tgt_lang")
        ]
        # target_prefix=[['__de__'], ['__fr__']]
        return self.model.translate_batch(*args, **kwds, target_prefix=target_prefix)

    def tokenize_encode(self, text, *args, **kwargs):
        tokens = []
        src_lang = kwargs.pop("src_lang")
        for t, src_language in zip(text, src_lang):
            self.tokenizer.src_lang = src_language
            tokens.append(
                self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(t))
            )
        return tokens

    def tokenize_decode(self, tokens_out, *args, **kwargs):
        return [
            self.tokenizer.decode(
                self.tokenizer.convert_tokens_to_ids(tokens_out[i].hypotheses[0][1:]),
                *args,
                **kwargs,
            )
            for i in range(len(tokens_out))
        ]

    def generate(
        self,
        text: Union[str, List[str]],
        src_lang: Union[str, List[str]],
        tgt_lang: Union[str, List[str]],
        *forward_args,
        **forward_kwds: Any,
    ):
        """_summary_

        Args:
            text (Union[str, List[str]]): Input texts
            src_lang (Union[str, List[str]]): soruce language of the Input texts
            tgt_lang (Union[str, List[str]]): target language for outputs
            max_batch_size (int, optional): Batch size. Defaults to 0.
            batch_type (str, optional): _. Defaults to "examples".
            asynchronous (bool, optional): Only False supported. Defaults to False.
            beam_size (int, optional): _. Defaults to 2.
            patience (float, optional): _. Defaults to 1.
            num_hypotheses (int, optional): _. Defaults to 1.
            length_penalty (float, optional): _. Defaults to 1.
            coverage_penalty (float, optional): _. Defaults to 0.
            repetition_penalty (float, optional): _. Defaults to 1.
            no_repeat_ngram_size (int, optional): _. Defaults to 0.
            disable_unk (bool, optional): _. Defaults to False.
            suppress_sequences (Optional[List[List[str]]], optional): _.
               Defaults to None.
            end_token (Optional[Union[str, List[str], List[int]]], optional): _.
               Defaults to None.
            return_end_token (bool, optional): _. Defaults to False.
            prefix_bias_beta (float, optional): _. Defaults to 0.
            max_input_length (int, optional): _. Defaults to 1024.
            max_decoding_length (int, optional): _. Defaults to 256.
            min_decoding_length (int, optional): _. Defaults to 1.
            use_vmap (bool, optional): _. Defaults to False.
            return_scores (bool, optional): _. Defaults to False.
            return_attention (bool, optional): _. Defaults to False.
            return_alternatives (bool, optional): _. Defaults to False.
            min_alternative_expansion_prob (float, optional): _. Defaults to 0.
            sampling_topk (int, optional): _. Defaults to 1.
            sampling_temperature (float, optional): _. Defaults to 1.
            replace_unknowns (bool, optional): _. Defaults to False.
            callback (_type_, optional): _. Defaults to None.

        Returns:
            Union[str, List[str]]: text as output, if list, same len as input
        """
        if not len(text) == len(src_lang) == len(tgt_lang):
            raise ValueError(
                f"unequal len: text={len(text)} src_lang={len(src_lang)} tgt_lang={len(tgt_lang)}"
            )
        forward_kwds["tgt_lang"] = tgt_lang
        return super().generate(
            text, *forward_args, **forward_kwds, encode_kwargs={"src_lang": src_lang}
        )


class EncoderCT2fromHfHub(CTranslate2ModelfromHuggingfaceHub):
    def __init__(
        self,
        model_name_or_path: str,
        device: Literal["cpu", "cuda"] = "cuda",
        device_index=0,
        compute_type: Literal["int8_float16", "int8"] = "int8_float16",
        tokenizer: Union[AutoTokenizer, None] = None,
        hub_kwargs={},
        **kwargs: Any,
    ):
        """for ctranslate2.Translator models, in particular m2m-100

        Args:
            model_name_or_path (str): _description_
            device (Literal[cpu, cuda], optional): _description_. Defaults to "cuda".
            device_index (int, optional): _description_. Defaults to 0.
            compute_type (Literal[int8_float16, int8], optional): _description_. Defaults to "int8_float16".
            tokenizer (Union[AutoTokenizer, None], optional): _description_. Defaults to None.
            hub_kwargs (dict, optional): _description_. Defaults to {}.
            **kwargs (Any, optional): Any additional arguments
        """
        self.ctranslate_class = ctranslate2.Encoder
        super().__init__(
            model_name_or_path,
            device,
            device_index,
            compute_type,
            tokenizer,
            hub_kwargs,
            **kwargs,
        )
        self.device = device
        if device == "cuda":
            try:
                import torch
            except ImportError:
                raise ValueError(
                    "decoding storageview on CUDA of encoder requires torch"
                )
            self.tensor_decode_method = functools.partial(
                torch.as_tensor, device=device
            )
            self.input_dtype = torch.int32
        else:
            try:
                import numpy as np
            except ImportError:
                raise ValueError(
                    "decoding storageview on CPU of encoder requires numpy"
                )
            self.tensor_decode_method = np.asarray

    def _forward(self, features, *args, **kwds):
        input_ids = features["input_ids"]
        tokens_out = self.model.forward_batch(input_ids, *args, **kwds)
        outputs = dict(
            pooler_output=self.tensor_decode_method(tokens_out.pooler_output),
            last_hidden_state=self.tensor_decode_method(tokens_out.last_hidden_state),
            attention_mask=features["attention_mask"],
        )
        return outputs

    def tokenize_encode(self, text, *args, **kwargs):
        return self.tokenizer(text)

    def tokenize_decode(self, tokens_out, *args, **kwargs):
        return tokens_out

    def generate(
        self,
        text: Union[str, List[str]],
        encode_tok_kwargs={},
        decode_tok_kwargs={},
        *forward_args,
        **forward_kwds: Any,
    ):
        return super().generate(
            text,
            encode_kwargs=encode_tok_kwargs,
            decode_kwargs=decode_tok_kwargs,
            *forward_args,
            **forward_kwds,
        )


class GeneratorCT2fromHfHub(CTranslate2ModelfromHuggingfaceHub):
    def __init__(
        self,
        model_name_or_path: str,
        device: Literal["cpu", "cuda"] = "cuda",
        device_index=0,
        compute_type: Literal["int8_float16", "int8"] = "int8_float16",
        tokenizer: Union[AutoTokenizer, None] = None,
        hub_kwargs={},
        **kwargs: Any,
    ):
        """for ctranslate2.Generator models

        Args:
            model_name_or_path (str): _description_
            device (Literal[cpu, cuda], optional): _description_. Defaults to "cuda".
            device_index (int, optional): _description_. Defaults to 0.
            compute_type (Literal[int8_float16, int8], optional): _description_. Defaults to "int8_float16".
            tokenizer (Union[AutoTokenizer, None], optional): _description_. Defaults to None.
            hub_kwargs (dict, optional): _description_. Defaults to {}.
            **kwargs (Any, optional): Any additional arguments
        """
        self.ctranslate_class = ctranslate2.Generator
        super().__init__(
            model_name_or_path,
            device,
            device_index,
            compute_type,
            tokenizer,
            hub_kwargs,
            **kwargs,
        )

    def _forward(self, *args, **kwds):
        return self.model.generate_batch(*args, **kwds)

    def tokenize_decode(self, tokens_out, *args, **kwargs):
        return [
            self.tokenizer.decode(tokens_out[i].sequences_ids[0], *args, **kwargs)
            for i in range(len(tokens_out))
        ]

    def generate(
        self,
        text: Union[str, List[str]],
        encode_tok_kwargs={},
        decode_tok_kwargs={},
        *forward_args,
        **forward_kwds: Any,
    ):
        """_summary_

        Args:
            text (str | List[str]): Input texts
            encode_tok_kwargs (dict, optional): additional kwargs for tokenizer
            decode_tok_kwargs (dict, optional): additional kwargs for tokenizer
            max_batch_size (int, optional): _. Defaults to 0.
            batch_type (str, optional): _. Defaults to 'examples'.
            asynchronous (bool, optional): _. Defaults to False.
            beam_size (int, optional): _. Defaults to 1.
            patience (float, optional): _. Defaults to 1.
            num_hypotheses (int, optional): _. Defaults to 1.
            length_penalty (float, optional): _. Defaults to 1.
            repetition_penalty (float, optional): _. Defaults to 1.
            no_repeat_ngram_size (int, optional): _. Defaults to 0.
            disable_unk (bool, optional): _. Defaults to False.
            suppress_sequences (Optional[List[List[str]]], optional): _.
                Defaults to None.
            end_token (Optional[Union[str, List[str], List[int]]], optional): _.
                Defaults to None.
            return_end_token (bool, optional): _. Defaults to False.
            max_length (int, optional): _. Defaults to 512.
            min_length (int, optional): _. Defaults to 0.
            include_prompt_in_result (bool, optional): _. Defaults to True.
            return_scores (bool, optional): _. Defaults to False.
            return_alternatives (bool, optional): _. Defaults to False.
            min_alternative_expansion_prob (float, optional): _. Defaults to 0.
            sampling_topk (int, optional): _. Defaults to 1.
            sampling_temperature (float, optional): _. Defaults to 1.

        Returns:
            str | List[str]: text as output, if list, same len as input
        """
        return super().generate(
            text,
            encode_kwargs=encode_tok_kwargs,
            decode_kwargs=decode_tok_kwargs,
            *forward_args,
            **forward_kwds,
        )
