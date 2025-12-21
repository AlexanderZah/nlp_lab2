from config import ORIGINAL_MODEL_NAME, QUANT_MODEL_NAME, QUANT_MODEL_NAME_TUNED, quantization_config
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import evaluate_subset, get_average_accuracy, get_model_size, load_mmlu_ds


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        QUANT_MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(QUANT_MODEL_NAME,
                                                 quantization_config=quantization_config,
                                                 trust_remote_code=True,
                                                 device_map="auto")

    model = PeftModel.from_pretrained(
        model, f"raler/{QUANT_MODEL_NAME_TUNED}", device_map="auto", torch_dtype="auto")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        f"raler/{QUANT_MODEL_NAME_TUNED}", trust_remote_code=True)

    average_accuracy = get_average_accuracy(model, tokenizer)
    print(f'Качество на бенчмарке MMLU: {average_accuracy:.2f}')
    model_size = get_model_size(model)
    print(f'Размер модели: {model_size:.4f}')


if __name__ == '__main__':
    main()
