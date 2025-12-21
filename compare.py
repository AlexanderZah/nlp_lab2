from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_average_accuracy, get_model_size
from config import ORIGINAL_MODEL_NAME, QUANT_MODEL_NAME, quantization_config

size_before_quant_mb = 0
size_after_quant_mb = 0
average_accuracy_before_quant = 0
average_accuracy_after_quant = 0


def get_original_model():
    tokenizer = AutoTokenizer.from_pretrained(
        ORIGINAL_MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        ORIGINAL_MODEL_NAME,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )

    size_before_quant_mb = get_model_size(model)
    print(f'Количество параметров ДО: {size_before_quant_mb:.2f} mb')

    average_accuracy_before_quant = get_average_accuracy(model, tokenizer)
    print(
        f'Качество на бенчмарке MMLU ДО: {average_accuracy_before_quant:.2f}')


def get_quant_model():
    model = AutoModelForCausalLM.from_pretrained(
        QUANT_MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        QUANT_MODEL_NAME, trust_remote_code=True)

    size_after_quant_mb = get_model_size(model)
    print(f'Количество параметров ПОСЛЕ: {size_after_quant_mb:.2f} mb')

    average_accuracy_after_quant = get_average_accuracy(model, tokenizer)
    print(
        f'Качество на бенчмарке MMLU ПОСЛЕ: {average_accuracy_after_quant:.2f}')


def main():
    get_original_model()

    get_quant_model()

    print('ИТОГ')
    print(f'Количество параметров ДО: {size_before_quant_mb:.2f} mb')
    print(f'Количество параметров ПОСЛЕ: {size_after_quant_mb:.2f} mb')
    print(
        f'Качество на бенчмарке MMLU ДО: {average_accuracy_before_quant:.2f}')
    # Качество на бенчмарке MMLU ДО: 0.65
    print(
        f'Качество на бенчмарке MMLU ПОСЛЕ: {average_accuracy_after_quant:.2f}')


if __name__ == '__main__':
    main()
