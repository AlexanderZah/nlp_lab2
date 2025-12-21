import torch
from torch import nn
from datasets import load_dataset, get_dataset_config_names
from tqdm import tqdm

from config import possible_choices, MAX_LENGTH, NUM_FEWSHOT, NUM_SUBSET


#### size ####
def get_model_size(model: nn.Module, unit='MB') -> float:
    total_bytes = 0
    for name, param in model.named_parameters():
        total_bytes += param.numel() * param.element_size()
    for buffer in model.buffers():
        total_bytes += buffer.numel() * buffer.element_size()
    if unit.upper() == 'MB':
        return total_bytes / (1024 * 1024)
    else:
        return total_bytes / (1024 * 1024 * 1024)  # GB


#### dataset ####

def load_mmlu_ds():
    all_subsets = get_dataset_config_names("cais/mmlu")
    all_subsets.remove("all")
    all_subsets.remove("auxiliary_train")

    dataset = {}
    for subset_name in all_subsets:
        dataset[subset_name] = load_dataset(
            "cais/mmlu", subset_name, split="test")

    return dataset


def format_example(subj, ex, include_answer=True):
    prompt = ex["question"]
    options = [ex["choices"][i] for i in range(len(possible_choices))]

    for i, option in enumerate(options):
        choice_letter = possible_choices[i]
        prompt += f"\n{choice_letter}. {option}"

    prompt += "\nAnswer:"

    if include_answer:
        correct_letter = possible_choices[ex['answer']]
        prompt += f" {correct_letter}\n\n"

    return prompt


def generate_fewshot_prompt(dev_data, subset_name, k):
    few_shot_prompt = ""
    indices = list(range(len(dev_data)))[:k]

    for idx in range(0, k):
        few_shot_prompt += format_example(subset_name, dev_data[idx])

    return few_shot_prompt


def get_index_from_choice(choice_letter):
    return possible_choices.index(choice_letter)


def evaluate_subset(model, subset_name, subset_data, tokenizer, verbose=False):

    correct_predictions = 0
    total_samples = 0

    target_token_ids = [tokenizer.convert_tokens_to_ids(
        c) for c in possible_choices]
    dev_data = load_dataset("cais/mmlu", subset_name, split="dev")
    for example in tqdm(subset_data):
        total_samples += 1
        if total_samples > NUM_SUBSET:
            break

        k = NUM_FEWSHOT

        few_shot_prompt = generate_fewshot_prompt(dev_data, subset_name, k)
        question_prompt = format_example(
            subset_name, example, include_answer=False)
        full_prompt = few_shot_prompt + question_prompt

        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True)
        input_ids = inputs.input_ids
        prompt_len = input_ids.shape[1]

        while prompt_len >= MAX_LENGTH:
            k -= 1
            few_shot_prompt = generate_fewshot_prompt(dev_data, subset_name, k)
            full_prompt = few_shot_prompt + question_prompt
            inputs = tokenizer(
                full_prompt, return_tensors="pt", truncation=True)
            input_ids = inputs.input_ids
            prompt_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        next_token_logits = logits[0, prompt_len - 1, :]

        log_probs = {}
        for i, token_id in enumerate(target_token_ids):
            choice_letter = possible_choices[i]
            score = next_token_logits[token_id].item()
            log_probs[choice_letter] = score

        predicted_choice = max(log_probs, key=log_probs.get)
        predicted_index = get_index_from_choice(predicted_choice)

        correct_index = example['answer']

        if predicted_index == correct_index:
            correct_predictions += 1

    accuracy = correct_predictions / total_samples
    if verbose:
        print(
            f"Точность для {subset_name}: {accuracy:.4f} ({correct_predictions}/{total_samples})")
    return accuracy, total_samples


def get_average_accuracy(model, tokenizer):
    dataset = load_mmlu_ds()
    all_accuracies = []
    all_samples = 0

    for subset_name, subset_data in dataset.items():
        acc, samples = evaluate_subset(
            model, subset_name, subset_data, tokenizer)
        all_accuracies.append(acc * samples)
        all_samples += samples

    average_accuracy = sum(all_accuracies) / all_samples

    return average_accuracy
