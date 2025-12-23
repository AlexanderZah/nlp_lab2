from config import ORIGINAL_MODEL_NAME, QUANT_MODEL_NAME, QUANT_MODEL_NAME_TUNED, quantization_config
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import evaluate_subset, get_average_accuracy, get_model_size, load_mmlu_ds

quant_model_tuned = None
quant_tokenizer = None


def main():
    global quant_tokenizer, quant_model_tuned
    quant_tokenizer = AutoTokenizer.from_pretrained(
        QUANT_MODEL_NAME, trust_remote_code=True)
    quant_model_tuned = AutoModelForCausalLM.from_pretrained(QUANT_MODEL_NAME,
                                                             quantization_config=quantization_config,
                                                             trust_remote_code=True,
                                                             device_map="auto")

    quant_model_tuned = PeftModel.from_pretrained(
        quant_model_tuned, f"raler/{QUANT_MODEL_NAME_TUNED}", device_map="auto", torch_dtype="auto")
    quant_model_tuned.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        f"raler/{QUANT_MODEL_NAME_TUNED}", trust_remote_code=True)

    average_accuracy = get_average_accuracy(quant_model_tuned, tokenizer)
    print(f'Качество на бенчмарке MMLU: {average_accuracy:.6f}')
    model_size = get_model_size(quant_model_tuned)
    print(f'Размер модели: {model_size:.6f} mb')
    # Качество на бенчмарке MMLU: 0.6550
    # Размер модели: 5920.59 mb


if __name__ == '__main__':
    main()
