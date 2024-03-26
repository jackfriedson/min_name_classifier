import re
import timeit


import numpy as np
import torch
from accelerate import Accelerator
from huggingface_hub import get_full_repo_name, Repository
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer


# Preprocessing

def tokenize_and_align_labels(tokenizer, examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def align_labels_with_tokens(labels: list[int], word_ids: list[int | None]) -> list[int]:
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


# Training and Evaluation

def compute_metrics(metric, label_names, eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


# Inference


word_split_regex = re.compile(r'\w+|[.,!?;]')


def split_into_words(input: str) -> list[str]:
    return word_split_regex.findall(input)


def align_predictions_with_words(predictions: list[int] | torch.Tensor, word_ids: list[int | None]) -> list[int]:
    if len(predictions) != len(word_ids):
        raise ValueError(
            f"Predictions and word_ids should have the same length, got {len(predictions)} predictions and {len(word_ids)} word_ids")

    word_predictions = []
    current_word = None
    for prediction, word_id in zip(predictions, word_ids):
        if word_id is None:
            continue
        if word_id != current_word:
            # Start of a new word
            word_predictions.append(prediction)
        current_word = word_id
    return word_predictions


class NER:
    def __init__(self, model, tokenizer, label_names, metric) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.metric = metric
        self.label_names = label_names

    @staticmethod
    def from_pretrained(model_checkpoint: str, label_names: list[str], metric) -> 'NER':
        id2label = {i: label for i, label in enumerate(label_names)}
        label2id = {v: k for k, v in id2label.items()}
        model = AutoModelForTokenClassification.from_pretrained(
            model_checkpoint,
            id2label=id2label,
            label2id=label2id,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        return NER(model, tokenizer, label_names, metric)

    def label_ids_to_names(self, label_ids: list[int]) -> list[str]:
        return [self.label_names[label_id] for label_id in label_ids if label_id != -100]

    def postprocess(self, predictions: torch.Tensor, labels: torch.Tensor) -> tuple[list[list[str]], list[list[str]]]:
        predictions = predictions.detach().cpu().clone().numpy()
        labels = labels.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[self.label_names[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [self.label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        return true_predictions, true_labels

    def evaluate_model(self, eval_dataloader: DataLoader) -> dict:
        self.model.eval()
        for batch in tqdm(eval_dataloader):
            with torch.no_grad():
                outputs = self.model(**batch)

            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]

            true_predictions, true_labels = self.postprocess(predictions, labels)
            self.metric.add_batch(predictions=true_predictions, references=true_labels)

        return self.metric.compute()

    def get_predictions(self, input: list[str]) -> list[str]:
        # Input must already be split into words
        tokenized_inputs = self.tokenizer(input, return_tensors="pt", is_split_into_words=True)
        with torch.no_grad():
            outputs = self.model(**tokenized_inputs)
        predictions = outputs.logits.argmax(dim=-1)
        word_predictions = align_predictions_with_words(predictions.tolist()[0], tokenized_inputs.word_ids())
        return self.label_ids_to_names(word_predictions)

    def get_predictions_batch(self, inputs: list[list[str]]) -> list[list[str]]:
        # Inputs must already be split into words
        tokenized_inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True,
                                          is_split_into_words=True)
        with torch.no_grad():
            outputs = self.model(**tokenized_inputs)
        predictions = outputs.logits.argmax(dim=-1)
        word_predictions = []
        for i in range(len(inputs)):
            word_prediction = align_predictions_with_words(predictions[i], tokenized_inputs.word_ids(batch_index=i))
            word_predictions.append(self.label_ids_to_names(word_prediction))
        return word_predictions

    def train(
            self,
            model_name: str,
            output_dir: str,
            num_train_epochs: int,
            train_dataloader: DataLoader,
            eval_dataloader: DataLoader,
            optimizer,
            lr_scheduler,
    ):
        accelerator = Accelerator()
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            self.model, optimizer, train_dataloader, eval_dataloader
        )

        repo_name = get_full_repo_name(model_name)
        repo = Repository(output_dir, clone_from=repo_name)

        for epoch in range(num_train_epochs):
            # Training
            model.train()
            for batch in tqdm(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Evaluation
            model.eval()
            for batch in tqdm(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)

                predictions = outputs.logits.argmax(dim=-1)
                labels = batch["labels"]

                # Necessary to pad predictions and labels for being gathered
                predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

                predictions_gathered = accelerator.gather(predictions)
                labels_gathered = accelerator.gather(labels)

                true_predictions, true_labels = self.postprocess(predictions_gathered, labels_gathered)
                self.metric.add_batch(predictions=true_predictions, references=true_labels)

            results = self.metric.compute()
            print(
                f"epoch {epoch}:",
                {
                    key: results[f"overall_{key}"]
                    for key in ["precision", "recall", "f1", "accuracy"]
                },
            )

            # Save and upload
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                self.tokenizer.save_pretrained(output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False
                )
