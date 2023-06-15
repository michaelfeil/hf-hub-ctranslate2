import os


def call(*args, **kwargs):
    import subprocess

    out = subprocess.call(*args, **kwargs)
    if out != 0:
        raise ValueError(f"Output: {out}")


model_description_generator = """
from hf_hub_ctranslate2 import GeneratorCT2fromHfHub
model = GeneratorCT2fromHfHub(
        # load in int8 on CUDA
        model_name_or_path=model_name,
        device="cuda",
        compute_type="int8_float16",
        # tokenizer=AutoTokenizer.from_pretrained("{ORG}/{NAME}")
)
outputs = model.generate(
    text=["def fibonnaci(", "User: How are you doing? Bot:"],
    max_length=64,
    include_prompt_in_result=False
)
print(outputs)"""

model_description_translator = """
from hf_hub_ctranslate2 import TranslatorCT2fromHfHub
model = TranslatorCT2fromHfHub(
        # load in int8 on CUDA
        model_name_or_path=model_name,
        device="cuda",
        compute_type="int8_float16",
        # tokenizer=AutoTokenizer.from_pretrained("{ORG}/{NAME}")
)
outputs = model.generate(
    text=["def fibonnaci(", "User: How are you doing? Bot:"],
    max_length=64,
)
print(outputs)"""

model_description_encoder = """
from hf_hub_ctranslate2 import EncoderCT2fromHfHub
model = EncoderCT2fromHfHub(
        # load in int8 on CUDA
        model_name_or_path=model_name,
        device="cuda",
        compute_type="float16",
        # tokenizer=AutoTokenizer.from_pretrained("{ORG}/{NAME}")
)
embeddings = model.generate(
    text=["I like soccer", "I like tennis", "The eiffel tower is in Paris"],
)
print(embeddings.shape, embeddings)
# getting correlation
embeddings_norm = embeddings / (embeddings**2).sum(axis=1, keepdims=True)**0.5
scores = (embeddings_norm @ embeddings_norm.T) * 100
"""


def convert(NAME="opus-mt-en-fr", ORG="Helsinki-NLP", description="generator"):
    print(f"converting {ORG}/{NAME} ")
    import re
    import datetime
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()

    HUB_NAME = f"ct2fast-{NAME}"
    repo_id = f"michaelfeil/{HUB_NAME}"
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
    tmp_dir = os.path.join(os.path.expanduser("~"), f"tmp-{HUB_NAME}")
    os.chdir(os.path.expanduser("~"))

    path = snapshot_download(
        f"{ORG}/{NAME}",
    )
    files = [f for f in os.listdir(path) if "." in f]
    filtered_f = [
        f
        for f in files
        if not ("model" in f or "config.json" == f or f.endswith(".py"))
    ]

    conv_arg = (
        [
            "ct2-transformers-converter",
            "--model",
            f"{ORG}/{NAME}",
            "--output_dir",
            str(tmp_dir),
            "--force",
            "--copy_files",
        ]
        + filtered_f
        + [
            "--quantization",
            "float16" if description == "encoder" else "int8_float16",
            "--trust_remote_code",
        ]
    )
    call(conv_arg)
    if not "vocabulary.txt" in os.listdir(tmp_dir) and "vocab.txt" in os.listdir(
        tmp_dir
    ):
        import shutil

        shutil.copyfile(
            os.path.join(tmp_dir, "vocab.txt"),
            os.path.join(tmp_dir, "vocabulary.txt"),
        )

    with open(os.path.join(tmp_dir, "README.md"), "r") as f:
        content = f.read()
    if "tags:" in content:
        content = content.replace("tags:", "tags:\n- ctranslate2\n- int8\n- float16", 1)
    else:
        content = content.replace(
            "---", "---\ntags:\n- ctranslate2\n- int8\n- float16\n", 1
        )

    end_header = [m.start() for m in re.finditer(r"---", content)]
    if len(end_header) > 1:
        end_header = end_header[1] + 3
    else:
        end_header = 0
    conv_arg_nice = " ".join(conv_arg)
    conv_arg_nice = conv_arg_nice.replace(os.path.expanduser("~"), "~")
    if description == "generator":
        model_description = model_description_generator
    elif description == "encoder":
        model_description = model_description_encoder
    elif description == "translator":
        model_description = model_description_translator
    add_string = f"""
# # Fast-Inference with Ctranslate2
Speedup inference while reducing memory by 2x-4x using int8 inference in C++ on CPU or GPU.

quantized version of [{ORG}/{NAME}](https://huggingface.co/{ORG}/{NAME})
```bash
pip install hf-hub-ctranslate2>=2.0.8 ctranslate2>=3.16.0
```
Converted on {str(datetime.datetime.now())[:10]} using
```
{conv_arg_nice}
```

Checkpoint compatible to [ctranslate2>=3.16.0](https://github.com/OpenNMT/CTranslate2)
and [hf-hub-ctranslate2>=2.0.8](https://github.com/michaelfeil/hf-hub-ctranslate2)
- `compute_type=int8_float16` for `device="cuda"`
- `compute_type=int8`  for `device="cpu"`

```python
from transformers import AutoTokenizer

model_name = "{repo_id}"
{model_description}
```

# Licence and other remarks:
This is just a quantized version. Licence conditions are intended to be idential to original huggingface repo.

# Original description
    """

    with open(os.path.join(tmp_dir, "README.md"), "w") as f:
        f.write(content[:end_header] + add_string + content[end_header:])

    api.upload_folder(
        folder_path=tmp_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload {ORG}/{NAME} ctranslate fp16 weights",
    )
    call(["rm", "-rf", tmp_dir])


if __name__ == "__main__":
    generators = [
        # "togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
        # "togethercomputer/GPT-JT-6B-v0",
        # "togethercomputer/RedPajama-INCITE-7B-Instruct",
        # "togethercomputer/RedPajama-INCITE-7B-Chat",
        # "EleutherAI/pythia-160m",
        # "EleutherAI/pythia-2.8b",
        # "EleutherAI/pythia-6.9b",
        # "EleutherAI/pythia-12b",
        # "togethercomputer/Pythia-Chat-Base-7B",
        # "stabilityai/stablelm-base-alpha-7b",
        # "stabilityai/stablelm-tuned-alpha-7b",
        # "stabilityai/stablelm-base-alpha-3b",
        # "stabilityai/stablelm-tuned-alpha-3b",
        # "OpenAssistant/stablelm-7b-sft-v7-epoch-3",
        # "EleutherAI/gpt-j-6b",
        # "EleutherAI/gpt-neox-20b",
        # "OpenAssistant/pythia-12b-sft-v8-7k-steps",
        # "Salesforce/codegen-350M-mono",
        # "Salesforce/codegen-350M-multi",
        # "Salesforce/codegen-2B-mono",
        # "Salesforce/codegen-2B-multi",
        # "Salesforce/codegen-6B-multi",
        # "Salesforce/codegen-6B-mono",
        # "Salesforce/codegen-16B-mono",
        # "Salesforce/codegen-16B-multi",
        # "Salesforce/codegen2-1B",
        # "Salesforce/codegen2-3_7B",
        # "Salesforce/codegen2-7B",
        # "Salesforce/codegen2-16B",
        # "bigcode/gpt_bigcode-santacoder",
        # 'bigcode/starcoder',
        # "mosaicml/mpt-7b",
        # "mosaicml/mpt-7b-instruct",
        # "mosaicml/mpt-7b-chat"
        "VMware/open-llama-7b-open-instruct",
        # "tiiuae/falcon-7b-instruct",
        # 'tiiuae/falcon-7b',
        "tiiuae/falcon-40b-instruct",
        "tiiuae/falcon-40b",
        "OpenAssistant/falcon-7b-sft-top1-696",
        "OpenAssistant/falcon-7b-sft-mix-2000",
        "OpenAssistant/falcon-40b-sft-mix-1226",
        # "HuggingFaceH4/starchat-beta",
        "WizardLM/WizardCoder-15B-V1.0",
    ]
    translators = [
        # 'Salesforce/codet5p-770m-py', 'Salesforce/codet5p-770m'
    ]
    encoders = [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "intfloat/e5-small-v2",
        "intfloat/e5-large-v2",
        "intfloat/e5-large",
        "sentence-transformers/all-MiniLM-L6-v2",
        "setu4993/LaBSE",
    ]
    for m in encoders:
        ORG, NAME = m.split("/")
        convert(NAME=NAME, ORG=ORG, description="encoder")

    for m in translators:
        ORG, NAME = m.split("/")
        convert(NAME=NAME, ORG=ORG, description="translator")

    for m in generators:
        ORG, NAME = m.split("/")
        # import huggingface_hub
        # huggingface_hub.snapshot_download(
        #     m
        # )
        convert(NAME=NAME, ORG=ORG, description="generator")

        from hf_hub_ctranslate2 import GeneratorCT2fromHfHub
        from transformers import AutoTokenizer

        model_name = f"michaelfeil/ct2fast-{NAME}"
        # use either TranslatorCT2fromHfHub or GeneratorCT2fromHfHub here, depending on model.
        model = GeneratorCT2fromHfHub(
            # load in int8 on CUDA
            model_name_or_path=model_name,
            device="cuda",
            compute_type="int8",
            tokenizer=AutoTokenizer.from_pretrained(m),
        )
        outputs = model.generate(
            text=["def print_hello_world():", "def hello_name(name:"], max_length=64
        )
        print(outputs)
