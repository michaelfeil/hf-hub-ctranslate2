import os

def call(*args, **kwargs):
    import subprocess
    out = subprocess.call(*args, **kwargs)
    if out != 0:
        raise ValueError(f"Output: {out}")

def convert(NAME="opus-mt-en-fr", ORG="Helsinki-NLP"):
    import re
    import datetime
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    
    HUB_NAME=f"ct2fast-{NAME}"
    repo_id = f"michaelfeil/{HUB_NAME}"
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
    tmp_dir = os.path.join(os.path.expanduser("~"), f"tmp-{HUB_NAME}")
    os.chdir(os.path.expanduser("~"))
    
    path = snapshot_download(
        f'{ORG}/{NAME}',
    )
    files = os.listdir(path)
    filtered_f = [f for f in files if not ("model" in f or "config.json" == f)]

    conv_arg = [
        'ct2-transformers-converter', 
        '--model',
        f'{ORG}/{NAME}',
        '--output_dir',
        str(tmp_dir),
        '--force',
        '--copy_files',
    ]+ filtered_f + [
        '--quantization',
        'float16']
    call(conv_arg)
    
    with open(os.path.join(tmp_dir,'README.md'),'r') as f:
        content = f.read()
    if "tags:" in content:
        content = content.replace("tags:","tags:\n- ctranslate2\n- int8\n- float16")
    else:
        content = content.replace("---","---\ntags:\n- ctranslate2\n- int8\n- float16\n")

    end_header = [m.start() for m in re.finditer(r"---",content)]
    if len(end_header) > 1:
        end_header = end_header[1] + 3
    else:
        end_header = 0
    conv_arg_nice = " ".join(conv_arg)
    add_string = f"""
# # Fast-Inference with Ctranslate2
Speedup inference while reducing memory by 2x-4x using int8 inference in C++ on CPU or GPU.

quantized version of [{ORG}/{NAME}](https://huggingface.co/{ORG}/{NAME})
```bash
pip install hf-hub-ctranslate2>=2.0.6 
```
Converted on {str(datetime.datetime.now())[:10]} using
```
{conv_arg_nice}
```

Checkpoint compatible to [ctranslate2>=3.13.0](https://github.com/OpenNMT/CTranslate2) and [hf-hub-ctranslate2>=2.0.6](https://github.com/michaelfeil/hf-hub-ctranslate2)
- `compute_type=int8_float16` for `device="cuda"` 
- `compute_type=int8`  for `device="cpu"`

```python
from hf_hub_ctranslate2 import TranslatorCT2fromHfHub, GeneratorCT2fromHfHub
from transformers import AutoTokenizer

model_name = "{repo_id}"
# use either TranslatorCT2fromHfHub or GeneratorCT2fromHfHub here, depending on model.
model = GeneratorCT2fromHfHub(
        # load in int8 on CUDA
        model_name_or_path=model_name, 
        device="cuda",
        compute_type="int8_float16",
        tokenizer=AutoTokenizer.from_pretrained("{ORG}/{NAME}")
)
outputs = model.generate(
    text=["How do you call a fast Flan-ingo?", "User: How are you doing? Bot:"],
)
print(outputs)
```

# Licence and other remarks:
This is just a quantized version. Licence conditions are intended to be idential to original huggingface repo.

# Original description
    """
    
    with open(os.path.join(tmp_dir,'README.md'),'w') as f:   
        f.write(content[:end_header] + add_string + content[end_header:])
    

    api.upload_folder(
        folder_path=tmp_dir,
        repo_id=repo_id,  repo_type="model",
        commit_message=f"Upload {ORG}/{NAME} ctranslate fp16 weights"
    )
    call(["rm","-rf", tmp_dir])
    
if __name__ == "__main__":
    generators = [
        ("togethercomputer/RedPajama-INCITE-Instruct-3B-v1"),
        ("togethercomputer/GPT-JT-6B-v0"),
        "togethercomputer/RedPajama-INCITE-Chat-7B-v0.1",
        "togethercomputer/RedPajama-INCITE-Instruct-7B-v0.1",
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-2.8b",
        "EleutherAI/pythia-6.9b",
        "EleutherAI/pythia-12b",
        "togethercomputer/Pythia-Chat-Base-7B",
        "stabilityai/stablelm-base-alpha-7b",
        "stabilityai/stablelm-tuned-alpha-7b",
        "stabilityai/stablelm-base-alpha-3b",
        "stabilityai/stablelm-tuned-alpha-3b",
        "OpenAssistant/stablelm-7b-sft-v7-epoch-3",
        "EleutherAI/gpt-j-6b",
        "EleutherAI/gpt-neox-20b",
        "OpenAssistant/pythia-12b-sft-v8-7k-steps"
    ]
    for m in generators:
        ORG , NAME = m.split("/")
        convert(NAME=NAME, ORG=ORG)
