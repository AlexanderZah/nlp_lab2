from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_average_accuracy, get_model_size
from config import ORIGINAL_MODEL_NAME, QUANT_MODEL_NAME, quantization_config

size_before_quant_mb = 0
size_after_quant_mb = 0
average_accuracy_before_quant = 0
average_accuracy_after_quant = 0
quant_model = None
quant_tokenizer = None
original_model = None
original_tokenizer = None


def get_original_model():
    global size_before_quant_mb, average_accuracy_before_quant, original_model, original_tokenizer

    original_tokenizer = AutoTokenizer.from_pretrained(
        ORIGINAL_MODEL_NAME, trust_remote_code=True)
    original_model = AutoModelForCausalLM.from_pretrained(
        ORIGINAL_MODEL_NAME,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )

    size_before_quant_mb = get_model_size(original_model)
    print(f'Количество параметров ДО: {size_before_quant_mb:.6f} mb')
    # Количество параметров ДО: 15622.59 mb
    average_accuracy_before_quant = get_average_accuracy(
        original_model, original_tokenizer)
    print(
        f'Качество на бенчмарке MMLU ДО: {average_accuracy_before_quant:.6f}')
    # Качество на бенчмарке MMLU ДО: 0.66


def get_quant_model():
    global size_after_quant_mb, average_accuracy_after_quant, quant_tokenizer, quant_model
    quant_model = AutoModelForCausalLM.from_pretrained(
        QUANT_MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )

    quant_tokenizer = AutoTokenizer.from_pretrained(
        QUANT_MODEL_NAME, trust_remote_code=True)

    size_after_quant_mb = get_model_size(quant_model)
    print(f'Количество параметров ПОСЛЕ: {size_after_quant_mb:.6f} mb')

    average_accuracy_after_quant = get_average_accuracy(
        quant_model, quant_tokenizer)
    print(
        f'Качество на бенчмарке MMLU ПОСЛЕ: {average_accuracy_after_quant:.6f}')


def main():
    get_original_model()

    get_quant_model()

    print('ИТОГ')
    print(f'Количество параметров ДО: {size_before_quant_mb:.6f} mb')
    # Количество параметров ДО: 15622.59 mb
    print(f'Количество параметров ПОСЛЕ: {size_after_quant_mb:.6f} mb')
    # Количество параметров ПОСЛЕ: 5686.59 mb
    print(
        f'Качество на бенчмарке MMLU ДО: {average_accuracy_before_quant:.6f}')
    # Качество на бенчмарке MMLU ДО: 0.6579
    print(
        f'Качество на бенчмарке MMLU ПОСЛЕ: {average_accuracy_after_quant:.6f}')
    # Качество на бенчмарке MMLU ПОСЛЕ: 0.6550
    Compression_ratio = size_before_quant_mb / size_after_quant_mb
    Performance_drop = (average_accuracy_before_quant -
                        average_accuracy_after_quant) / average_accuracy_before_quant
    Score = Compression_ratio / (1 + Performance_drop)
    print(
        f'Score: {Score:.6f}')
    '''
    ИТОГ
    Количество параметров ДО: 15622.59 mb
    Количество параметров ПОСЛЕ: 5686.59 mb
    Качество на бенчмарке MMLU ДО: 0.6579
    Качество на бенчмарке MMLU ПОСЛЕ: 0.6550
    Score: 2.74
    '''


if __name__ == '__main__':
    main()
