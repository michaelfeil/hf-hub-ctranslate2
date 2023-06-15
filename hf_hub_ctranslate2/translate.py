import ctranslate2

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
            model_path = model_name_or_path
        else:
            try:
                model_path = _utils._download_model(
                    model_name_or_path, hub_kwargs=hub_kwargs
                )
            except:
                hub_kwargs["local_files_only"] = True
                model_path = _utils._download_model(
                    model_name_or_path, hub_kwargs=hub_kwargs
                )
        self.model = self.ctranslate_class(
            model_path,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            **kwargs,
        )

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            if "tokenizer.json" in os.listdir(model_path):
                if not autotokenizer_ok:
                    raise ValueError(
                        "`pip install transformers` missing to load AutoTokenizer."
                    )
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, fast=True)
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
            [self.tokenizer.lang_code_to_token[l]] for l in kwds.pop("tgt_lang")
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
            import functools

            try:
                import torch
            except ImportError:
                raise ValueError(
                    "decoding storageview on CUDA of encoder requires torch"
                )
            self.tensor_decode_method = functools.partial(
                torch.as_tensor, device=device
            )
        else:
            try:
                import numpy as np
            except ImportError:
                raise ValueError(
                    "decoding storageview on CPU of encoder requires numpy"
                )
            self.tensor_decode_method = np.asarray

    def _forward(self, *args, **kwds):
        return self.model.forward_batch(*args, **kwds)

    def tokenize_encode(self, text, *args, **kwargs):
        return self.tokenizer(text).input_ids

    def tokenize_decode(self, tokens_out, *args, **kwargs):
        return self.tensor_decode_method(tokens_out.pooler_output)

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

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = False,
        *args,
        **kwargs,
    ):
        """
        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        import numpy as np

        if convert_to_tensor:
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(
            sentences, "__len__"
        ):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in range(0, len(sentences), batch_size):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]

            embeddings = self.generate(sentences_batch)

            if normalize_embeddings:
                embeddings = (
                    embeddings / (embeddings**2).sum(axis=1, keepdims=True) ** 0.5
                )

            # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
            if convert_to_numpy and not isinstance(embeddings, np.ndarray):
                embeddings = embeddings.cpu()

            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor and not isinstance(all_embeddings[0], np.ndarray):
            raise NotImplementedError
        elif convert_to_numpy and not isinstance(all_embeddings[0], np.ndarray):
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings


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
