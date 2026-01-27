# katabatic/models/great/great_start.py

import random
import numpy as np
import typing as tp


def _pad(x, length: int, pad_value: int):
    """Prepend pad_value until x reaches length."""
    return [pad_value] * (length - len(x)) + x


def _pad_tokens(tokens, pad_value: int):
    """
    Pads tokenized sequences in a list so all have same length.
    """
    max_length = len(max(tokens, key=len))
    tokens = [_pad(t, max_length, pad_value=pad_value) for t in tokens]
    return tokens


class GReaTStart:
    """
    Abstract base class for creating start tokens for generation.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        # Use tokenizer pad_token_id if available, else fallback to eos_token_id, else 0.
        self.pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else (tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)
        )

    def get_start_tokens(self, n_samples: int) -> tp.List[tp.List[int]]:
        raise NotImplementedError("Must be implemented by subclasses.")


class CategoricalStart(GReaTStart):
    def __init__(self, tokenizer, start_col: str, start_col_dist: dict):
        super().__init__(tokenizer)
        assert isinstance(start_col, str)
        assert isinstance(start_col_dist, dict)

        self.start_col = start_col
        self.population = list(start_col_dist.keys())
        self.weights = list(start_col_dist.values())

    def get_start_tokens(self, n_samples: int):
        start_words = random.choices(self.population, self.weights, k=n_samples)
        start_text = [f"{self.start_col} is {s}," for s in start_words]
        tokenized = self.tokenizer(start_text)["input_ids"]
        return _pad_tokens(tokenized, pad_value=self.pad_token_id)


class ContinuousStart(GReaTStart):
    def __init__(
        self,
        tokenizer,
        start_col: str,
        start_col_dist: tp.List[float],
        noise: float = 0.01,
        decimal_places: int = 5,
    ):
        super().__init__(tokenizer)
        assert isinstance(start_col, str)
        assert isinstance(start_col_dist, list)

        self.start_col = start_col
        self.start_col_dist = start_col_dist
        self.noise = noise
        self.decimal_places = decimal_places

    def get_start_tokens(self, n_samples: int):
        start_words = random.choices(self.start_col_dist, k=n_samples)

        # optional: add noise (commented as in your code)
        # start_words = np.array(start_words) + np.random.normal(size=n_samples) * self.noise

        start_text = [
            f"{self.start_col} is {format(s, f'.{self.decimal_places}f')},"
            for s in start_words
        ]
        tokenized = self.tokenizer(start_text)["input_ids"]
        return _pad_tokens(tokenized, pad_value=self.pad_token_id)


class RandomStart(GReaTStart):
    def __init__(self, tokenizer, all_columns: tp.List[str]):
        super().__init__(tokenizer)
        self.all_columns = all_columns

    def get_start_tokens(self, n_samples: int):
        start_words = random.choices(self.all_columns, k=n_samples)
        start_text = [f"{s} is " for s in start_words]
        tokenized = self.tokenizer(start_text)["input_ids"]
        return _pad_tokens(tokenized, pad_value=self.pad_token_id)
