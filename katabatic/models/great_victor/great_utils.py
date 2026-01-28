import typing as tp

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer


def _array_to_dataframe(data: tp.Union[pd.DataFrame, np.ndarray], columns=None) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data

    assert isinstance(data, np.ndarray), "Input must be a Pandas DataFrame or a Numpy NDArray"
    assert columns, "If data is ndarray, you must pass a list of column names"
    assert len(columns) == len(data[0]), (
        f"{len(columns)} column names given, but array has {len(data[0])} columns!"
    )

    return pd.DataFrame(data=data, columns=columns)


def _get_column_distribution(df: pd.DataFrame, col: str) -> tp.Union[list, dict]:
    """
    If numeric => return list of values
    else => categorical distribution dict {value: prob}
    """
    if pd.api.types.is_numeric_dtype(df[col]):
        return df[col].tolist()
    return df[col].value_counts(normalize=True, dropna=False).to_dict()


def _convert_tokens_to_text(
    tokens: tp.Union[tp.List[torch.Tensor], torch.Tensor, tp.List[tp.List[int]]],
    tokenizer: AutoTokenizer,
) -> tp.List[str]:
    """
    Robust decoding for outputs from generate():
    - accepts list[tensor], tensor, or list[list[int]]
    """
    if isinstance(tokens, torch.Tensor):
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        decoded = tokenizer.batch_decode(tokens, skip_special_tokens=True)
        return [d.replace("\n", " ").replace("\r", "").strip() for d in decoded]

    norm = []
    for t in tokens:
        if isinstance(t, torch.Tensor):
            norm.append(t.detach().cpu().tolist())
        else:
            norm.append(t)
    decoded = tokenizer.batch_decode(norm, skip_special_tokens=True)
    return [d.replace("\n", " ").replace("\r", "").strip() for d in decoded]


def _convert_text_to_tabular_data(text: tp.List[str], columns: tp.List[str]) -> pd.DataFrame:
    """
    Convert generated "col is value, col2 is value2, ..." into dataframe.
    """
    generated = []

    for t in text:
        features = t.split(",")
        td = dict.fromkeys(columns, "placeholder")

        for f in features:
            f = f.strip()
            if " is " not in f:
                continue
            left, right = f.split(" is ", 1)
            left = left.strip()
            right = right.strip()

            if left in columns and td[left] == "placeholder":
                if right == "" or right.lower() == "none":
                    td[left] = None
                else:
                    td[left] = right

        generated.append(td)

    df_gen = pd.DataFrame(generated)
    df_gen.replace("None", None, inplace=True)
    return df_gen


def _encode_row_partial(row, shuffle=True, float_precision=None):
    num_cols = len(row.index)
    idx_list = np.arange(num_cols) if not shuffle else np.random.permutation(num_cols)

    parts = []
    for i in idx_list:
        value = row[row.index[i]]
        if pd.isna(value):
            continue

        col_name = row.index[i]
        if isinstance(value, (float, np.floating)) and float_precision is not None:
            s = f"{float(value):.{float_precision}f}"
            if "." in s:
                s = s.rstrip("0").rstrip(".")
            final = s
        else:
            final = value

        parts.append(f"{col_name} is {final}")

    return ", ".join(parts)


def _get_random_missing(row):
    nans = list(row[pd.isna(row)].index)
    return np.random.choice(nans) if len(nans) > 0 else None


def _partial_df_to_promts(partial_df: pd.DataFrame, float_precision=None):
    encoder = lambda x: _encode_row_partial(x, True, float_precision)
    res_encode = list(partial_df.apply(encoder, axis=1))
    res_first = list(partial_df.apply(_get_random_missing, axis=1))

    return [
        ((enc + ", ") if len(enc) > 0 else "") + (fst + " is" if fst is not None else "")
        for enc, fst in zip(res_encode, res_first)
    ]


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
