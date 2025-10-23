from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import exist_and_not_none, zero_pad_sequences
import numpy as np

cnt = 0

def preprocess_data(
    data,
    input_template=None,
    context_key="context_messages",
    label_key="label",
    strength_key="strength",
    apply_chat_template=None,
    is_scaled=False,
) -> str:
    # context = [{
    #     "content": "Compare the quality of two responses from the assistant to the user. The two conversations are provided in order.", 
    #     "role": "system"
    # }] + data[context_key]
    context = data[context_key]
    if apply_chat_template:
        context = apply_chat_template(context, tokenize=False)
    else:
        raise "Preference data init not implemented."

    # margin loss
    label = data[label_key] * 1.
    if is_scaled: 
        # strength = data[strength_key]
        strength = float(data[strength_key])
    else:
        strength = 1.

    return context, label, strength


class PreferenceDataset(Dataset):
    """
    Dataset for preference model

    Args:
        dataset: dataset for preference model
        self.tokenizer: self.tokenizer for preference model
        self.max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        is_dpo=False,
        num_processors=8,
        multiple_of=1,
    ) -> None:
        super().__init__()
        self.is_dpo = is_dpo
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.multiple_of = multiple_of
        # print('>>>>> Loss type in strategy: ', self.strategy.args.loss)
        self.is_scaled = (self.strategy.args.loss == "scaled_bt")

        # chat_template
        self.input_template = input_template
        self.prompt_key = getattr(self.strategy.args, "prompt_key", None)
        self.context_key = getattr(self.strategy.args, "context_key", None)
        self.label_key = getattr(self.strategy.args, "label_key", None)
        self.strength_key = getattr(self.strategy.args, "strength_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # Parallel loading datasets
        processed_dataset = dataset.map(
            self.process_data, remove_columns=dataset.column_names, num_proc=num_processors
        )

        # Filter out None values if necessary
        # processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        # Store the processed data in class attributes
        self.contexts = processed_dataset["context"]
        self.labels = processed_dataset["label"]
        self.strengths = processed_dataset["strength"]

    def process_data(self, data):
        context, label, strength = preprocess_data(
            data,
            self.input_template,
            self.context_key,
            self.label_key,
            self.strength_key,
            self.apply_chat_template,
            self.is_scaled,
        )

        if self.is_dpo:
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

            # Filter the sample whose length is greater than max_length (2 for answer length)
            if prompt_ids_len >= self.max_length - 2:
                prompt = None

        return {
            "context": context,
            "label": label,
            "strength": strength,
        }

    def __len__(self):
        length = len(self.contexts)
        return length

    def __getitem__(self, idx):
        context, label, strength = self.contexts[idx], self.labels[idx], self.strengths[idx]

        chosen = context.rstrip("\n")
        if not chosen.endswith(self.tokenizer.eos_token):
            chosen += " " + self.tokenizer.eos_token
        chosen_token = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        # to avoid EOS_token truncation
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id[-1] if isinstance(self.tokenizer.eos_token_id, list) else self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True

        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            label,
            strength
        )

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        labels = []
        strengths = []
        for chosen_id, chosen_mask, label, strength in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            labels.append(label)
            strengths.append(strength)

        if self.is_dpo:
            padding_side = "right"
        else:
            padding_side = "left"
        chosen_ids = zero_pad_sequences(chosen_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        chosen_masks = zero_pad_sequences(chosen_masks, side=padding_side)
        return chosen_ids, chosen_masks, labels, strengths

    def packing_collate_fn(self, item_list):
        labels = []
        strengths = []

        chosen_ids = []
        chosen_att_masks = []
        chosen_seq_lens = []
        index = 1
        for chosen_id, chosen_mask, label, strength in item_list:
            chosen_ids.append(chosen_id.flatten())
            chosen_att_masks.append(torch.full_like(chosen_id.flatten(), index))
            chosen_seq_lens.append(len(chosen_id.flatten()))
            labels.append(label)
            strengths.append(strength)

            index += 1

        packed_input_ids = torch.cat(chosen_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(chosen_att_masks, dim=0).unsqueeze(0)
        packed_seq_lens = chosen_seq_lens

        if self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0:
            padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value=0)

        return packed_input_ids, packed_attention_masks, packed_seq_lens, labels, strengths
