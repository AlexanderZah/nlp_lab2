from config import ORIGINAL_MODEL_NAME, quantization_config
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import evaluate_subset, get_average_accuracy, get_model_size, load_mmlu_ds

tokenizer = AutoTokenizer.from_pretrained(
    ORIGINAL_MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(ORIGINAL_MODEL_NAME,
                                             quantization_config=quantization_config,
                                             trust_remote_code=True,
                                             device_map="auto")

model = PeftModel.from_pretrained(model, "raler/qwen3-8b-qlora-finetuned")
model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    "raler/qwen3-8b-qlora-finetuned", trust_remote_code=True)

average_accuracy = get_average_accuracy(model, tokenizer)
print(f'Качество на бенчмарке MMLU: {average_accuracy:.2f}')
model_size = get_model_size(model)
print('Размер модели: {model_size:.4f}')
