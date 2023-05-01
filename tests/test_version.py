from hf_hub_ctranslate2 import __version__


def test_version():
    assert isinstance(__version__, str)
    assert len(__version__.split(".")) == 3
