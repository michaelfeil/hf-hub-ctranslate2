import ctranslate2
import os

model_id = "declare-lab/flan-alpaca-xl"
output_dir = os.path.join(os.path.dirname(__file__), model_id.split("/")[-1])


class FastTokenizerDrop(ctranslate2.converters.TransformersConverter):
    def load_model(self, model_class, model_name_or_path, **kwargs):
        kwargs["low_cpu_mem_usage"] = True
        return super().load_model(model_class, model_name_or_path, **kwargs)

    def load_tokenizer(self, tokenizer_class, model_name_or_path, **kwargs):
        # FlanAlpaca ships without a fast version of the tokenizer
        # This is a workaround to avoid the error about it missing in the model files
        del kwargs["use_fast"]
        return super().load_tokenizer(tokenizer_class, model_name_or_path, **kwargs)


# Use extended Converter class as a workaround for the missing fast tokenizer
# and to reduce RAM usage during conversion
ct = FastTokenizerDrop(model_name_or_path=model_id, load_as_float16=True)
ct.convert(output_dir=output_dir, force=True, quantization="int8_float16")
