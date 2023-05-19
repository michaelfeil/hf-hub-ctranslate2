from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

version = " . "
with open("hf_hub_ctranslate2/__init__.py", "r", encoding="utf-8") as fh:
    lines = fh.readlines()
    for l in lines[::-1]:
        if "__version__" in l and "=" in l:
            print(l)
            version = l.split("__version__")[-1]
            version = version.replace("=", "").replace("'", "").replace('"', "").strip()
            break
if len(version.split(".")) != 3:
    raise ValueError(f"Version incorrect: {version}")


setup(
    name="hf_hub_ctranslate2",
    packages=find_packages(),
    version=version,
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
        "ctranslate2>=3.13.0",
        "transformers>=4.28.0",
        "huggingface-hub",
        "typing_extensions",
    ],
)
