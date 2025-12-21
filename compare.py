import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils import get_average_accuracy, get_model_size, evaluate_subset, load_mmlu_ds
from config import ORIGINAL_MODEL_NAME, QUANT_MODEL_NAME, quantization_config


tokenizer = AutoTokenizer.from_pretrained(
    ORIGINAL_MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    ORIGINAL_MODEL_NAME,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype="auto"
)


size_before_quant_mb = get_model_size(model)
print(f'Количество параметров ДО: {size_before_quant_mb:.2f}')

average_accuracy = get_average_accuracy(model, tokenizer)
print(f'Качество на бенчмарке MMLU ДО: {average_accuracy:.2f}')

print()

model = AutoModelForCausalLM.from_pretrained(
    QUANT_MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    QUANT_MODEL_NAME, trust_remote_code=True)

size_before_quant_mb = get_model_size(model)
print(f'Количество параметров ПОСЛЕ: {size_before_quant_mb:.2f}')

average_accuracy = get_average_accuracy(model, tokenizer)
print(f'Качество на бенчмарке MMLU ПОСЛЕ: {average_accuracy:.2f}')
