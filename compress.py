from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_model_size
from config import ORIGINAL_MODEL_NAME, quantization_config

quant_model = None
tokenizer = None


def main():
    global quant_model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        ORIGINAL_MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(ORIGINAL_MODEL_NAME,
                                                 quantization_config=quantization_config,
                                                 trust_remote_code=True,
                                                 device_map="auto")
    model.eval()

    model_size = get_model_size(model)
    print(f'Размер модели: {model_size:.6f} mb')


if __name__ == '__main__':
    main()
