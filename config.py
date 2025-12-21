from transformers import BitsAndBytesConfig
import torch

ORIGINAL_MODEL_NAME = "Qwen/Qwen3-8B"
QUANT_MODEL_NAME = "raler/compressed_model_BitsAndBytesConfig_4bit"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

possible_choices = ['A', 'B', 'C', 'D']
NUM_SUBSET = 5
NUM_FEWSHOT = 2
MAX_LENGTH = 1024
