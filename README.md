# Лабораторная работа: Компрессия и дообучение LLM (Qwen3-8B)

**Модель:** [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)  
**Бенчмарк:** MMLU (Multi-task Multiple-choice Language Understanding)  
**Датасет:** `cais/mmlu` (Hugging Face Datasets)  
**Оценка:**  
- Compression ratio = Original_size / Compressed_size  
- Performance drop = (Original_metric - Compressed_metric) / Original_metric  
- Score = Compression_ratio / (1 + Performance_drop)  

## Описание лабораторной работы

### Этап 1 — Компрессия модели (пост-трейнинг)


В данной реализации используется **4-битная квантизация NF4 + double quantization** (BitsAndBytes).

### Этап 2 — Дообучение сжатой модели

Результаты дообучения хранятся в репозитории `raler/qwen3-8b-qlora-finetuned1`.

## Структура репозитория

| Файл                  | Описание                                                                                   |
|-----------------------|--------------------------------------------------------------------------------------------|
| `config.py`           | Основные конфигурации: имя модели, параметры квантизации, константы для MMLU (кол-во few-shot, max length и т.д.) |
| `utils.py`            | Вспомогательные функции: <br>• загрузка MMLU датасета <br>• форматирование примеров <br>• оценка качества (evaluate_subset) <br>• подсчёт размера модели (get_model_size) |
| `compress.py`         | Скрипт для загрузки и квантизации исходной модели (сохраняет результат в `QUANT_MODEL_NAME`) |
| `compare.py`          | Сравнивает исходную и сжатую модель: <br>• размер до и после <br>• качество на MMLU до и после |
| `Inference.py`        | Загружает сжатую + дообученную модель (PEFT/LoRA) и замеряет качество + размер |
| `train.py`            | Дообучение сжатой модели (QLoRA + LoRA) на подмножестве MMLU auxiliary_train |

## Используемые технологии

- **Базовая модель:** Qwen/Qwen3-8B  
- **Квантизация (Этап 1):** `bitsandbytes` (4-bit NF4 + double quantization)  
- **Дообучение (Этап 2):** `peft` (LoRA) + `trl` (SFTTrainer)  
- **Оценка:** MMLU (test split, few-shot 2, подмножество 5 примеров на категорию для скорости)  

## Как запустить

### Этап 1 — Компрессия

```bash
# Сжать модель (загрузит и сохранит квантизованную версию)
python compress.py
```
#### Сравнение до/после
```bash
# Замерить размер и качество на MMLU до и после квантизации
python compare.py
```
### Этап 2 — Дообучение
```bash
# Запустить обучение (QLoRA + LoRA)
python train.py
```
#### Инференс дообученной модели
```bash
# Загрузить сжатую + дообученную модель и получить метрики
python inference.py
```
