import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


model_name = "Qwen/Qwen3-8B"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             quantization_config=quantization_config,
                                             trust_remote_code=True,
                                             device_map="auto")
model.eval()
