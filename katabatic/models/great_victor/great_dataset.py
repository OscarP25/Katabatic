# katabatic/models/great/great_dataset.py

import random
import typing as tp
from dataclasses import dataclass

import numpy as np
from datasets import Dataset as HFDataset
from transformers import DataCollatorWithPadding


class GReaTDataset:
    """
    Wrapper around a HuggingFace Dataset that:
      - permutes columns per row
      - converts row to "col is value, ..." text
      - tokenizes (WITHOUT padding; collator pads)
    """

    def __init__(self, hf_dataset: HFDataset):
        self.ds = hf_dataset
        self.tokenizer = None
        self.float_precision = None
        self.column_names = hf_dataset.column_names

    @classmethod
    def from_pandas(cls, df):
        hf = HFDataset.from_pandas(df, preserve_index=False)
        return cls(hf)

    def set_tokenizer(self, tokenizer, float_precision=None):
        self.tokenizer = tokenizer
        self.float_precision = float_precision

    def _format_value(self, value) -> str:
        if isinstance(value, (float, np.floating)) and self.float_precision is not None:
            s = f"{float(value):.{self.float_precision}f}"
            if "." in s:
                s = s.rstrip("0").rstrip(".")
            return s
        return str(value).strip()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> tp.Dict[str, tp.Any]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not set. Call set_tokenizer(...) before training.")

        row = self.ds[idx]  # dict: {col: value}

        # shuffle feature order
        cols = self.column_names.copy()
        random.shuffle(cols)

        # build prompt text
        parts = []
        for c in cols:
            parts.append(f"{c} is {self._format_value(row[c])}")
        text = ", ".join(parts)

        # IMPORTANT: no padding here; collator handles it
        tokenized = self.tokenizer(text, padding=False, truncation=True)
        return tokenized


@dataclass
class GReaTDataCollator(DataCollatorWithPadding):
    """
    Pads inputs and sets labels = input_ids for causal LM fine-tuning.
    """

    def __call__(self, features: tp.List[tp.Dict[str, tp.Any]]):
        batch = self.tokenizer.pad(
            features,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch
