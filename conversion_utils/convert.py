import os
import json
import shutil
import subprocess
import re
import datetime
    
def call(*args, **kwargs):
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
        compute_type="int8_float16"
)
outputs = model.generate(
    text=["I like soccer", "I like tennis", "The eiffel tower is in Paris"],
    max_length=64,
) # perform downstream tasks on outputs
outputs["pooler_output"]
outputs["last_hidden_state"]
outputs["attention_mask"]

# alternative, use SentenceTransformer Mix-In
# for end-to-end Sentence embeddings generation
# (not pulling from this CT2fast-HF repo)

from hf_hub_ctranslate2 import CT2SentenceTransformer
model = CT2SentenceTransformer(
    model_name_orig, compute_type="int8_float16", device="cuda"
)
embeddings = model.encode(
    ["I like soccer", "I like tennis", "The eiffel tower is in Paris"],
    batch_size=32,
    convert_to_numpy=True,
    normalize_embeddings=True,
)
print(embeddings.shape, embeddings)
scores = (embeddings @ embeddings.T) * 100
"""


def convert(NAME="opus-mt-en-fr", ORG="Helsinki-NLP", description="generator"):
    print(f"converting {ORG}/{NAME} ")

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()

    HUB_NAME = f"ct2fast-{NAME}"
    repo_id = f"michaelfeil/{HUB_NAME}"
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
    tmp_dir = os.path.join(os.path.expanduser("~"), f"tmp-{HUB_NAME}")
    os.chdir(os.path.expanduser("~"))

    path = snapshot_download(
        f"{ORG}/{NAME}",
        ignore_patterns=("*.bin", "*.safetensors)"),
        resume_download=True,
    )
    files = [f for f in os.listdir(path) if "." in f]
    filtered_f = [
        f
        for f in files
        if not ("model" in f or "config.json" == f or f.endswith(".py"))
    ]
    if "config.json" in files:
        with open(os.path.join(path,"config.json"),"r") as f:
            transformers_config = json.load(f)

    # conv_arg = (
    #     [
    #         "ct2-transformers-converter",
    #         "--model",
    #         f"{ORG}/{NAME}",
    #         "--output_dir",
    #         str(tmp_dir),
    #         "--force",
    #         "--copy_files",
    #     ]
    #     + filtered_f
    #     + ([
    #         "--quantization", "int8_float16",
    #     ] if description != "encoder" else [])
    #     + [
    #         "--trust_remote_code",
    #     ]
    # )
    # call(conv_arg)
    from ctranslate2.converters import TransformersConverter
    converter = TransformersConverter(
        f"{ORG}/{NAME}",
        activation_scales=None,
        copy_files=filtered_f,
        load_as_float16=True,
        revision=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    converter.convert(
        output_dir=str(tmp_dir),
        vmap = None, # TODO: vmap here
        quantization="int8_float16"  if description != "encoder" else None,
        force = True,
    )
    
    if not "vocabulary.txt" in os.listdir(tmp_dir) and "vocab.txt" in os.listdir(
        tmp_dir
    ):
        shutil.copyfile(
            os.path.join(tmp_dir, "vocab.txt"),
            os.path.join(tmp_dir, "vocabulary.txt"),
        )
    if not "vocabulary.txt" in os.listdir(tmp_dir) and "vocabulary.json" in os.listdir(
        tmp_dir
    ):
        with open(os.path.join(tmp_dir,"vocabulary.json"),"r") as f:
            vocab = json.load(f)
        with open(os.path.join(tmp_dir,"vocabulary.txt"),"w") as f:
            f.write('\n'.join(str(i) for i in vocab))
            
        
    if "config.json" in os.listdir(tmp_dir):
        with open(os.path.join(tmp_dir,"config.json"),"r") as f:
            ct2_config = json.load(f)
        
        new_config = {
            **transformers_config,
            **ct2_config
        }
        with open(os.path.join(tmp_dir,"config.json"),"w") as f:
            json.dump(new_config, f, indent=4)

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
    # conv_arg_nice = " ".join(conv_arg)
    conv_arg_nice = "LLama-2 -> removed <pad> token."
    # conv_arg_nice = conv_arg_nice.replace(os.path.expanduser("~"), "~")
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
pip install hf-hub-ctranslate2>=2.12.0 ctranslate2>=3.17.1
```

```python
# from transformers import AutoTokenizer
model_name = "{repo_id}"
{('model_name_orig="'+ORG+"/"+NAME + '"') if description == "encoder" else ""}
{model_description}
```

Checkpoint compatible to [ctranslate2>=3.17.1](https://github.com/OpenNMT/CTranslate2)
and [hf-hub-ctranslate2>=2.12.0](https://github.com/michaelfeil/hf-hub-ctranslate2)
- `compute_type=int8_float16` for `device="cuda"`
- `compute_type=int8`  for `device="cpu"`

Converted on {str(datetime.datetime.now())[:10]} using
```
{conv_arg_nice}
```

# Licence and other remarks:
This is just a quantized version. Licence conditions are intended to be idential to original huggingface repo.

# Original description
    """
    fp = os.path.join(tmp_dir, "model.bin")
    if os.stat(f"{fp}").st_size > 15_000_000_500:
        # in chunks for 9GB
        call([f"split","-d","-b","9GB",f"{fp}",f"{fp}-"])
        call([f"rm",f"{fp}"])
        all_chunks = list(sorted([f for f in os.listdir(tmp_dir) if f"model.bin-" in f]))
        max_chunk_num = all_chunks[-1].split("-")[-1]
        for ck in all_chunks:
            chunk_num = ck.split("-")[-1]
            shutil.move(os.path.join(tmp_dir,ck), os.path.join(tmp_dir, f"model-{chunk_num}-of-{max_chunk_num}.bin"))

    with open(os.path.join(tmp_dir, "README.md"), "w") as f:
        f.write(content[:end_header] + add_string + content[end_header:])

    api.upload_folder(
        folder_path=tmp_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload {ORG}/{NAME} ctranslate2 weights",
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
        # "VMware/open-llama-7b-open-instruct",
        # "VMware/open-llama-13b-open-instruct",
        # "tiiuae/falcon-7b-instruct",
        # 'tiiuae/falcon-7b',
        # "tiiuae/falcon-40b",
        # "tiiuae/falcon-40b-instruct",
        # "OpenAssistant/falcon-7b-sft-top1-696",
        # "OpenAssistant/falcon-7b-sft-mix-2000",
        # "togethercomputer/GPT-JT-6B-v1",
        # "OpenAssistant/falcon-40b-sft-mix-1226",
        # "HuggingFaceH4/starchat-beta",
        # "WizardLM/WizardCoder-15B-V1.0",
        # "mosaicml/mpt-30b-instruct",
        # "mosaicml/mpt-30b",
        # "mosaicml/mpt-30b-chat",
        # "Salesforce/xgen-7b-8k-base",
        # "Salesforce/xgen-7b-8k-inst",
        # "Salesforce/codegen25-7b-multi",
        # "Salesforce/codegen25-7b-mono",
        # "Salesforce/codegen25-7b-instruct",
        # "meta-llama/Llama-2-7b-hf",
        # "meta-llama/Llama-2-7b-chat-hf",
        # "meta-llama/Llama-2-13b-hf",
        # "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/Llama-2-70b-chat-hf",
        "meta-llama/Llama-2-70b-hf",
    ]
    translators = [
        # 'Salesforce/codet5p-770m-py', 'Salesforce/codet5p-770m',
        # "facebook/nllb-200-distilled-1.3B",
        # "facebook/nllb-200-3.3B",
    ]
    encoders = [
        # "intfloat/e5-small-v2",
        # "intfloat/e5-small",
        # "intfloat/e5-large-v2",
        # "intfloat/e5-large",
        # "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        # "sentence-transformers/all-MiniLM-L6-v2",
        # "sentence-transformers/all-MiniLM-L12-v2",
        # "setu4993/LaBSE",
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

        # from hf_hub_ctranslate2 import GeneratorCT2fromHfHub
        # from transformers import AutoTokenizer

        # model_name = f"michaelfeil/ct2fast-{NAME}"
        # # use either TranslatorCT2fromHfHub or GeneratorCT2fromHfHub here, depending on model.
        # model = GeneratorCT2fromHfHub(
        #     # load in int8 on CUDA
        #     model_name_or_path=model_name,
        #     device="cpu",
        #     compute_type="int8",
        #     tokenizer=AutoTokenizer.from_pretrained(m),
        # )
        # outputs = model.generate(
        #     text=["def print_hello_world():", "def hello_name(name:"], max_length=64
        # )
        # print(outputs)
