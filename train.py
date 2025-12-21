import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

from config import QUANT_MODEL_NAME, quantization_config


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

    # ... (форматирование данных, как в предыдущем примере)

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
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",  # если вы добавили поле text
        max_seq_length=2048,
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
