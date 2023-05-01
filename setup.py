from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath("__file__")))
import hf_hub_ctranslate2

setup(
    name="hf_hub_ctranslate2",
    packages=find_packages(),
    version=hf_hub_ctranslate2.__version__,
    description=("Connecting Transfromers on HuggingfaceHub with Ctranslate2 "),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Michael Feil",
    license="MIT",
    url="https://github.com/michaelfeil/hf-hub-ctranslate2",
    project_urls={
        "Bug Tracker": "https://github.com/michaelfeil/hf-hub-ctranslate2/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    install_requires=[
        "ctranslate2>=3.13",
        "transformers>=4.28.*",
        "huggingface-hub",
    ],
)
