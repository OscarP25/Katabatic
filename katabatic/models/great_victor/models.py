# katabatic/models/great/models.py
"""
Katabatic integration for GReaT (be_great-style) tabular generation.

This file is designed to work with Katabatic's Pipeline interface:
- train(dataset_dir: str, synthetic_dir: str, **kwargs) must exist
- writes: synthetic_dir/x_synth.csv and synthetic_dir/y_synth.csv

Safe defaults are chosen to avoid crashes on CPU-only machines / VS Code:
- small LLM suggested: "distilgpt2"
- low epochs/batch_size
- capped synthetic rows via max_synth
"""

from __future__ import annotations

import os
import re
import json
import random
import logging
import warnings
import typing as tp

import numpy as np
import pandas as pd
import fsspec
import torch

# Katabatic base model
from katabatic.models.base_model import Model

# Local GReaT utilities (your folder: katabatic/models/great/)
from .great_dataset import GReaTDataset, GReaTDataCollator
from .great_start import (
    GReaTStart,
    CategoricalStart,
    ContinuousStart,
    RandomStart,
    _pad_tokens,
)
from .great_trainer import GReaTTrainer
from .great_utils import (
    _array_to_dataframe,
    _get_column_distribution,
    _convert_tokens_to_text,
    _convert_text_to_tabular_data,
    _partial_df_to_promts,
    bcolors,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GReaT(Model):
    """
    GReaT model wrapper for Katabatic.

    Notes:
    - Requires: transformers, datasets, torch, tqdm, fsspec
    - For Katabatic pipeline training, call:
        model.train(dataset_dir=<...>, synthetic_dir=<...>, **kwargs)
    """

    def __init__(
        self,
        llm: str = "distilgpt2",
        experiment_dir: str = "trainer_great",
        epochs: int = 1,
        batch_size: int = 1,
        efficient_finetuning: str = "",
        float_precision: tp.Optional[int] = None,
        report_to: tp.List[str] = [],
        **train_kwargs,
    ):
        super().__init__()
        self.check_dependencies()

        # Lazy import transformers so the module import doesn't crash on missing deps
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: F401
        except Exception as e:
            raise ImportError(
                "GReaT requires HuggingFace 'transformers'. Install with:\n"
                "  pip install transformers datasets tqdm fsspec\n"
                "Also ensure torch is installed.\n"
                f"Original error: {e}"
            )

        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.efficient_finetuning = efficient_finetuning
        self.llm = llm

        # Tokenizer / model
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm)
        # make sure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.llm)

        # Optional LoRA
        if self.efficient_finetuning == "lora":
            try:
                from peft import (
                    LoraConfig,
                    get_peft_model,
                    prepare_model_for_int8_training,
                    TaskType,
                )
            except ImportError:
                raise ImportError(
                    "LoRA requested but 'peft' is not installed. Install:\n"
                    "  pip install peft==0.9.0"
                )

            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["c_attn"],  # GPT2-specific
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = prepare_model_for_int8_training(self.model)
            self.model = get_peft_model(self.model, lora_config)
            try:
                self.model.print_trainable_parameters()
            except Exception:
                pass

        # Training params
        self.experiment_dir = experiment_dir
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.train_hyperparameters = {"report_to": report_to, **train_kwargs}

        # Sampling metadata
        self.columns: tp.Optional[list[str]] = None
        self.num_cols: tp.Optional[list[str]] = None
        self.conditional_col: tp.Optional[str] = None
        self.conditional_col_dist: tp.Optional[tp.Union[dict, list]] = None

        self.float_precision = float_precision

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        # used by Katabatic Model.check_dependencies()
        return ["torch", "transformers", "datasets", "tqdm", "fsspec"]

    # -------------------------
    # Katabatic expected method
    # -------------------------
    def train(self, dataset_dir: str, synthetic_dir: str, **kwargs):
        """
        Katabatic pipeline entrypoint.

        Expects:
          dataset_dir/x_train.csv
          dataset_dir/y_train.csv   (single column)

        Writes:
          synthetic_dir/x_synth.csv
          synthetic_dir/y_synth.csv
        """
        logger.info("=" * 80)
        logger.info("Training GReaT (Katabatic pipeline mode)")
        logger.info(f"dataset_dir={dataset_dir}")
        logger.info(f"synthetic_dir={synthetic_dir}")
        logger.info("=" * 80)

        x_train_path = os.path.join(dataset_dir, "x_train.csv")
        y_train_path = os.path.join(dataset_dir, "y_train.csv")
        if not os.path.exists(x_train_path):
            raise FileNotFoundError(f"Missing: {x_train_path}")
        if not os.path.exists(y_train_path):
            raise FileNotFoundError(f"Missing: {y_train_path}")

        X_train = pd.read_csv(x_train_path)
        y_train_df = pd.read_csv(y_train_path)

        if y_train_df.shape[1] != 1:
            raise ValueError(
                f"Expected y_train.csv to have 1 column, got {y_train_df.shape[1]}"
            )

        y_name = y_train_df.columns[0]
        y_train = y_train_df.iloc[:, 0]

        df_train = pd.concat([X_train, y_train], axis=1)

        # -------------------------
        # SAFE CAPS (prevent crash)
        # -------------------------
        # allow passing from runner/pipeline
        max_synth = kwargs.get("max_synth", None)
        sample_k = int(kwargs.get("sample_k", 16))
        sample_max_length = int(kwargs.get("sample_max_length", 160))
        sample_temperature = float(kwargs.get("sample_temperature", 0.7))
        device = kwargs.get("device", "cpu")  # force cpu by default
        guided_sampling = bool(kwargs.get("guided_sampling", False))
        random_feature_order = bool(kwargs.get("random_feature_order", True))
        drop_nan = bool(kwargs.get("drop_nan", False))

        # Extra safety: keep fine-tuning tiny unless explicitly small already
        if self.epochs > 2:
            self.epochs = 1
        if self.batch_size > 2:
            self.batch_size = 1

        # Fit / finetune
        self.fit(df_train)

        # Decide how many rows to generate
        n_rows = len(df_train)
        if max_synth is not None:
            try:
                n_rows = min(n_rows, int(max_synth))
            except Exception:
                pass

        logger.info(f"Generating synthetic rows: {n_rows} (device={device})")

        df_synth = self.sample(
            n_samples=n_rows,
            temperature=sample_temperature,
            k=sample_k,
            max_length=sample_max_length,
            drop_nan=drop_nan,
            device=device,
            guided_sampling=guided_sampling,
            random_feature_order=random_feature_order,
        )

        # Split X/y (label is last column)
        if df_synth.shape[1] < 2:
            # fallback
            x_synth = df_synth.copy()
            y_synth = pd.Series([0] * len(x_synth), name=y_name)
        else:
            x_synth = df_synth.iloc[:, :-1].copy()
            y_synth = df_synth.iloc[:, -1].copy()
            y_synth.name = y_name

        # Align columns
        if x_synth.shape[1] == X_train.shape[1]:
            x_synth.columns = X_train.columns
            x_synth = x_synth.reindex(columns=X_train.columns)

        # Convert to numeric where possible (discretized datasets often ints)
        for c in x_synth.columns:
            x_synth[c] = pd.to_numeric(x_synth[c], errors="ignore")
        y_synth = pd.to_numeric(y_synth, errors="ignore")

        # Save
        os.makedirs(synthetic_dir, exist_ok=True)
        x_synth.to_csv(os.path.join(synthetic_dir, "x_synth.csv"), index=False)
        pd.DataFrame(y_synth, columns=[y_name]).to_csv(
            os.path.join(synthetic_dir, "y_synth.csv"), index=False
        )

        logger.info(f"Saved x_synth/y_synth to: {synthetic_dir}")
        return self

    def evaluate(self, *args, **kwargs):
        # pipeline handles evaluation
        return None

    # -------------------------
    # be_great-style API
    # -------------------------
    def fit(
        self,
        data: tp.Union[pd.DataFrame, np.ndarray],
        column_names: tp.Optional[tp.List[str]] = None,
        conditional_col: tp.Optional[str] = None,
        resume_from_checkpoint: tp.Union[bool, str] = False,
    ) -> GReaTTrainer:
        from transformers import TrainingArguments

        df = _array_to_dataframe(data, columns=column_names)
        self._update_column_information(df)
        self._update_conditional_information(df, conditional_col)

        logger.info("Convert data into HuggingFace dataset object...")
        great_ds = GReaTDataset.from_pandas(df)
        great_ds.set_tokenizer(self.tokenizer, self.float_precision)

        logger.info("Create GReaT Trainer...")
        training_args = TrainingArguments(
            self.experiment_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            **self.train_hyperparameters,
        )

        great_trainer = GReaTTrainer(
            self.model,
            training_args,
            train_dataset=great_ds,
            tokenizer=self.tokenizer,
            data_collator=GReaTDataCollator(self.tokenizer),
        )

        logger.info("Start training...")
        great_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        return great_trainer

    def sample(
        self,
        n_samples: int,
        start_col: tp.Optional[str] = "",
        start_col_dist: tp.Optional[tp.Union[dict, list]] = None,
        temperature: float = 0.7,
        k: int = 100,
        max_length: int = 100,
        drop_nan: bool = False,
        device: str = "cpu",
        guided_sampling: bool = False,
        random_feature_order: bool = True,
    ) -> pd.DataFrame:
        if guided_sampling:
            return self._guided_sample(
                n_samples=n_samples,
                temperature=temperature,
                max_length=max_length,
                device=device,
                random_feature_order=random_feature_order,
            )
        return self._legacy_sample(
            n_samples=n_samples,
            start_col=start_col,
            start_col_dist=start_col_dist,
            temperature=temperature,
            k=k,
            max_length=max_length,
            drop_nan=drop_nan,
            device=device,
        )

    def _guided_sample(
        self,
        n_samples: int = 10,
        temperature: float = 0.7,
        max_length: int = 100,
        device: str = "cpu",
        random_feature_order: bool = True,
    ) -> pd.DataFrame:
        if self.columns is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        # Move model to device
        self.model.to(device)
        self.model.eval()

        synthetic_data: list[dict[str, tp.Any]] = []

        # Use tqdm if available (imported in other module); avoid hard dependency here
        try:
            from tqdm import tqdm
            iterator = tqdm(range(n_samples), total=n_samples)
        except Exception:
            iterator = range(n_samples)

        for i in iterator:
            try:
                feature_names = self.columns.copy()
                if random_feature_order:
                    random.shuffle(feature_names)

                sample_text = ""
                sample_values: dict[str, str] = {}

                for feature in feature_names:
                    prompt = f"{sample_text}{feature} is"
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

                    # Generate a short continuation (value)
                    out = self.model.generate(
                        inputs["input_ids"],
                        max_length=len(inputs["input_ids"][0]) + min(30, max_length),
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                    generated_text = self.tokenizer.decode(out[0], skip_special_tokens=True)
                    raw_value = generated_text[len(prompt):].strip()

                    # Parse value
                    if ";" in raw_value:
                        value = raw_value.split(";")[0].strip()
                    else:
                        # stop at commas/newlines first
                        for delim in [",", "\n"]:
                            if delim in raw_value:
                                value = raw_value.split(delim)[0].strip()
                                break
                        else:
                            value = raw_value[:30].strip()

                    # strip trailing junk
                    while value and not (value[-1].isalnum() or value[-1] in [".", "-"]):
                        value = value[:-1]

                    # numeric hint
                    if self.num_cols and feature in self.num_cols:
                        m = re.search(r"-?\d+\.?\d*", value)
                        if m:
                            value = m.group(0)

                    sample_values[feature] = value
                    sample_text += f"{feature} is {value}; "

                synthetic_data.append({f: sample_values.get(f, "") for f in self.columns})
            except Exception as e:
                logger.warning(f"Guided sample failed at {i+1}: {e}")
                continue

        if not synthetic_data:
            return pd.DataFrame(columns=self.columns)

        df = pd.DataFrame(synthetic_data)

        # convert numeric cols
        if self.num_cols:
            for col in self.num_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.head(n_samples)

    def _legacy_sample(
        self,
        n_samples: int,
        start_col: tp.Optional[str] = "",
        start_col_dist: tp.Optional[tp.Union[dict, list]] = None,
        temperature: float = 0.7,
        k: int = 100,
        max_length: int = 100,
        drop_nan: bool = False,
        device: str = "cpu",
    ) -> pd.DataFrame:
        great_start = self._get_start_sampler(start_col, start_col_dist)

        self.model.to(device)
        self.model.eval()

        dfs: list[pd.DataFrame] = []
        already_generated = 0
        attempts = 0

        try:
            from tqdm import tqdm
            pbar = tqdm(total=n_samples)
        except Exception:
            pbar = None

        while already_generated < n_samples:
            attempts += 1
            start_tokens = great_start.get_start_tokens(k)
            start_tokens = torch.tensor(start_tokens).to(device)

            tokens = self.model.generate(
                input_ids=start_tokens,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            text_data = _convert_tokens_to_text(tokens, self.tokenizer)
            df_gen = _convert_text_to_tabular_data(text_data, self.columns)

            # clean
            df_gen = df_gen[~(df_gen == "placeholder").any(axis=1)]
            df_gen = df_gen.dropna(how="all")
            if drop_nan:
                df_gen = df_gen.dropna()

            # numeric cols: coerce
            if self.num_cols:
                for c in self.num_cols:
                    coerced = pd.to_numeric(df_gen[c], errors="coerce")
                    df_gen = df_gen[coerced.notnull() | df_gen[c].isna()]
                df_gen[self.num_cols] = df_gen[self.num_cols].apply(
                    pd.to_numeric, errors="coerce"
                )

            dfs.append(df_gen)
            already_generated += len(df_gen)
            if pbar is not None:
                pbar.update(len(df_gen))

            # stop if stuck
            if attempts > 15 and already_generated == 0:
                raise RuntimeError(
                    "Unable to generate samples after multiple attempts. "
                    "Try guided_sampling=True or increase epochs/max_length."
                )

        if pbar is not None:
            pbar.close()

        df_out = pd.concat(dfs, ignore_index=True)
        return df_out.head(n_samples)

    def great_sample(
        self,
        starting_prompts: tp.Union[str, list[str]],
        temperature: float = 0.7,
        max_length: int = 100,
        device: str = "cpu",
    ) -> pd.DataFrame:
        self.model.to(device)
        self.model.eval()

        prompts = [starting_prompts] if isinstance(starting_prompts, str) else starting_prompts
        generated_tokens = []

        for prompt in prompts:
            start_token = torch.tensor(self.tokenizer(prompt)["input_ids"]).to(device)
            gen = self.model.generate(
                input_ids=torch.unsqueeze(start_token, 0),
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            generated_tokens.append(torch.squeeze(gen))

        decoded = _convert_tokens_to_text(generated_tokens, self.tokenizer)
        return _convert_text_to_tabular_data(decoded, self.columns)

    def impute(
        self,
        df_miss: pd.DataFrame,
        temperature: float = 0.7,
        k: int = 100,
        max_length: int = 100,
        max_retries: int = 15,
        device: str = "cpu",
    ) -> pd.DataFrame:
        if self.columns is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        if set(df_miss.columns) != set(self.columns):
            raise ValueError("Columns of df_miss must match training columns.")

        self.model.to(device)
        self.model.eval()

        out_rows = []
        for idx in range(len(df_miss)):
            df_curr = df_miss.iloc[[idx]].copy()
            org_index = df_curr.index
            retries = 0

            while retries < max_retries:
                prompts = _partial_df_to_promts(df_curr, self.float_precision)
                df_gen = self.great_sample(prompts, temperature, max_length, device=device)

                # coerce numeric columns
                if self.num_cols:
                    for c in self.num_cols:
                        df_gen[c] = pd.to_numeric(df_gen[c], errors="coerce")

                if not df_gen.isna().any().any():
                    out_rows.append(df_gen.set_index(org_index))
                    break

                retries += 1

            if retries == max_retries:
                warnings.warn("Max retries reached during imputation; row may contain NaNs.")
                out_rows.append(df_gen.set_index(org_index))

        return pd.concat(out_rows, axis=0)

    def save(self, path: str):
        fs = fsspec.filesystem(fsspec.utils.get_protocol(path))
        if fs.exists(path):
            warnings.warn(f"Directory {path} already exists; overwriting.")
        else:
            fs.mkdir(path)

        with fs.open(path + "/config.json", "w") as f:
            attributes = self.__dict__.copy()
            attributes.pop("tokenizer", None)
            attributes.pop("model", None)

            if isinstance(attributes.get("conditional_col_dist", None), np.ndarray):
                attributes["conditional_col_dist"] = list(attributes["conditional_col_dist"])

            json.dump(attributes, f)

        torch.save(self.model.state_dict(), fs.open(path + "/model.pt", "wb"))

    def load_finetuned_model(self, path: str):
        self.model.load_state_dict(torch.load(fsspec.open(path, "rb")))

    @classmethod
    def load_from_dir(cls, path: str):
        fs = fsspec.filesystem(fsspec.utils.get_protocol(path))
        assert fs.exists(path), f"Directory {path} does not exist."

        with fs.open(path + "/config.json", "r") as f:
            attributes = json.load(f)

        great = cls(attributes.get("llm", "distilgpt2"))
        for k, v in attributes.items():
            setattr(great, k, v)

        great.model.load_state_dict(
            torch.load(fs.open(path + "/model.pt", "rb"), map_location="cpu")
        )
        return great

    # -------------------------
    # internal helpers
    # -------------------------
    def _update_column_information(self, df: pd.DataFrame):
        self.columns = df.columns.to_list()
        self.num_cols = df.select_dtypes(include=np.number).columns.to_list()

    def _update_conditional_information(self, df: pd.DataFrame, conditional_col: tp.Optional[str] = None):
        if conditional_col is not None:
            if not isinstance(conditional_col, str):
                raise TypeError("conditional_col must be a string or None.")
            if conditional_col not in df.columns:
                raise ValueError(f"conditional_col '{conditional_col}' not in df.columns")

        self.conditional_col = conditional_col if conditional_col else df.columns[-1]
        self.conditional_col_dist = _get_column_distribution(df, self.conditional_col)

    def _get_start_sampler(
        self,
        start_col: tp.Optional[str],
        start_col_dist: tp.Optional[tp.Union[tp.Dict, tp.List]],
    ) -> GReaTStart:
        if start_col and start_col_dist is None:
            raise ValueError(f"start_col '{start_col}' given but start_col_dist is None.")
        if start_col_dist is not None and not start_col:
            raise ValueError("start_col_dist given but start_col is missing.")

        start_col = start_col if start_col else self.conditional_col
        start_col_dist = start_col_dist if start_col_dist else self.conditional_col_dist

        if isinstance(start_col_dist, dict):
            return CategoricalStart(self.tokenizer, start_col, start_col_dist)
        if isinstance(start_col_dist, list):
            return ContinuousStart(self.tokenizer, start_col, start_col_dist)
        return RandomStart(self.tokenizer, self.columns)


# Katabatic convention: export model name used by imports
GReaTModel = GReaT
