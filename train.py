import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from torch.optim import AdamW

from config import QUANT_MODEL_NAME, quantization_config


def format_example_for_training(example):
    """
    Адаптируем вашу функцию format_example для обучения:
    - Добавляем вопрос + варианты + правильный ответ.
    - Без subject, т.к. он не нужен в промпте.
    """
    prompt = example["question"]
    options = example["choices"]

    for i, option in enumerate(options):
        choice_letter = chr(65 + i)  # A, B, C, D
        prompt += f"\n{choice_letter}. {option}"

    prompt += "\nAnswer:"

    # Добавляем правильный ответ
    correct_letter = chr(65 + example["answer"])
    prompt += f" {correct_letter}\n\n"

    return {"text": prompt}


def main():

    model = AutoModelForCausalLM.from_pretrained(
        QUANT_MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        QUANT_MODEL_NAME, trust_remote_code=True)

    # Подготавливаем модель для QLoRA (важно!)
    model = prepare_model_for_kbit_training(model)

    # LoRA-конфиг
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # для Qwen
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    # Датасет (пример — MMLU auxiliary_train)
    dataset = load_dataset("cais/mmlu", "all", split='auxiliary_train')

    dataset = dataset.map(format_example_for_training, num_proc=4)

    # Тренировка
    training_args = TrainingArguments(
        output_dir="qwen3-8b-qlora-finetuned",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        max_grad_norm=0.3,
        num_train_epochs=3,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        fp16=False,
        optim="paged_adamw_32bit",
        gradient_checkpointing=True,  # отключаем — убирает ошибку
        report_to="none",

    )
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]

    optimizer = AdamW(lora_params, lr=2e-4)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        optimizers=(optimizer, None)
    )

    trainer.train()

    # Сохраняем только LoRA-адаптер (маленький!)
    trainer.model.save_pretrained("qwen3-8b-qlora-finetuned")
    tokenizer.save_pretrained("qwen3-8b-qlora-finetuned")

    # Можно сразу запушить на HF
    trainer.model.push_to_hub("raler/qwen3-8b-qlora-finetuned")
    tokenizer.push_to_hub("raler/qwen3-8b-qlora-finetuned")


if __name__ == '__main__':
    main()
