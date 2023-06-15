from typing import Optional
import huggingface_hub

from tqdm.auto import tqdm


def _download_model(
    model_name: str,
    output_dir: Optional[str] = None,
    local_files_only: bool = False,
    cache_dir: Optional[str] = None,
    hub_kwargs={},
):
    """Downloads a CTranslate2 model from the Hugging Face Hub.
    # adaptions from https://github.com/guillaumekln/faster-whisper

    Args:
      model_name: repo name on HF Hub e.g.  "michaelfeil/ct2fast-flan-alpaca-base"
      output_dir: Directory where the model should be saved. If not set,
         the model is saved in  the cache directory.
      local_files_only:  If True, avoid downloading the file and return the
        path to the local  cached file if it exists.
      cache_dir: Path to the folder where cached files are stored.

    Returns:
      The path to the downloaded model.

    Raises:
      ValueError: if the model size is invalid.
    """

    kwargs = hub_kwargs
    kwargs["local_files_only"] = local_files_only
    if output_dir is not None:
        kwargs["local_dir"] = output_dir
        kwargs["local_dir_use_symlinks"] = False

    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir

    allow_patterns = [
        "config.json",
        "model.bin",
        "tokenizer.json",
        "vocabulary.txt",
        "tokenizer_config.json",
        "*ocabulary.txt",
        "vocab.txt",
    ]

    return huggingface_hub.snapshot_download(
        model_name,
        allow_patterns=allow_patterns,
        tqdm_class=_disabled_tqdm,
        **kwargs,
    )


class _disabled_tqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)
