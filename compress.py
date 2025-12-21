from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_model_size
from config import ORIGINAL_MODEL_NAME, quantization_config


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        ORIGINAL_MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(ORIGINAL_MODEL_NAME,
                                                 quantization_config=quantization_config,
                                                 trust_remote_code=True,
                                                 device_map="auto")
    model.eval()

    model_size = get_model_size(model)
    print(f'Размер модели: {model_size:.4f}')


if __name__ == '__main__':
    main()
