from hf_hub_ctranslate2 import TranslatorCT2fromHfHub, GeneratorCT2fromHfHub, MultiLingualTranslatorCT2fromHfHub
from hf_hub_ctranslate2.util import utils as _utils
from transformers import AutoTokenizer


def test_translator(model_name="michaelfeil/ct2fast-flan-alpaca-base"):
    model = TranslatorCT2fromHfHub(
        model_name_or_path=model_name, device="cpu", compute_type="int8"
    )

    outputs = model.generate(
        ["How do you call a fast Flan-ingo?", "Translate to german: How are you doing?"]
    )
    assert len(outputs) == 2
    assert len(outputs[0]) != len(outputs[1])
    assert "flan" in outputs[0].lower()
    for o in outputs:
        assert isinstance(o, str)

def test_multilingualtranslator(model_name="michaelfeil/ct2fast-m2m100_418M"):
    model = MultiLingualTranslatorCT2fromHfHub(
        model_name_or_path=model_name, device="cpu", compute_type="int8",
        tokenizer=AutoTokenizer.from_pretrained(f"facebook/{model_name.split('-')[-1]}")
    )

    outputs = model.generate(
        ["How do you call a fast Flamingo?", "Wie geht es dir?"],
        src_lang=["en", "de"],
        tgt_lang=["de", "fr"]
    )
    assert len(outputs) == 2
    assert len(outputs[0]) != len(outputs[1])
    assert "nennt" in outputs[0].lower()
    assert "comment" in outputs[1].lower()
    for o in outputs:
        assert isinstance(o, str)

def test_generator(model_name="michaelfeil/ct2fast-pythia-160m"):
    model = GeneratorCT2fromHfHub(
        model_name_or_path=model_name, device="cpu", compute_type="int8"
    )

    outputs = model.generate(
        ["How do you call a fast Flan-ingo?", "Translate to german: How are you doing?"]
    )
    assert len(outputs) == 2
    assert len(outputs[0]) != len(outputs[1])
    assert "flan" in outputs[0].lower()
    for o in outputs:
        assert isinstance(o, str)


def test_generator_single(model_name="michaelfeil/ct2fast-pythia-160m"):
    model_path = _utils._download_model(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = GeneratorCT2fromHfHub(
        model_name_or_path=model_path,
        device="cpu",
        compute_type="int8",
        tokenizer=tokenizer,
    )

    outputs = model.generate("How do you call a fast Flan-ingo?")
    assert isinstance(outputs, str)
    assert "flan" in outputs.lower()


if __name__ == "__main__":
    test_generator()
    test_translator()
