import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection
from typing import Optional


class DataPrep(object):

    def __init__(
        self,
        raw_df: pd.DataFrame,
        categorical: list,
        log: list,
        mixed: dict,
        general: list,
        non_categorical: list,
        integer: list,
        type: dict,
        test_ratio: Optional[float]
    ):

        self.categorical_columns = categorical
        self.log_columns = log
        self.mixed_columns = mixed
        self.general_columns = general
        self.non_categorical_columns = non_categorical
        self.integer_columns = integer

        self.column_types = {
            "categorical": [],
            "mixed": {},
            "general": [],
            "non_categorical": []
        }

        self.lower_bounds = {}
        self.label_encoder_list = []

        
        # Handle supervised / unsupervised setting
        
        problem = list(type.keys())[0] if type else None
        target_col = list(type.values())[0] if type else None

        if problem:
            y_real = raw_df[target_col]
            X_real = raw_df.drop(columns=[target_col])

            if problem == "Classification":
                X_train_real, _, y_train_real, _ = model_selection.train_test_split(
                    X_real,
                    y_real,
                    test_size=test_ratio,
                    stratify=y_real,
                    random_state=42
                )
            else:
                X_train_real, _, y_train_real, _ = model_selection.train_test_split(
                    X_real,
                    y_real,
                    test_size=test_ratio,
                    random_state=42
                )

            X_train_real[target_col] = y_train_real
            self.df = X_train_real
        else:
            self.df = raw_df.copy()

        
        # Missing value handling
        
        self.df = self.df.replace(r' ', np.nan)
        self.df = self.df.fillna('empty')

        all_columns = set(self.df.columns)
        irrelevant_missing_columns = set(self.categorical_columns)
        relevant_missing_columns = list(all_columns - irrelevant_missing_columns)

        for col in relevant_missing_columns:
            if col in self.log_columns:
                if "empty" in self.df[col].values:
                    self.df[col] = self.df[col].apply(
                        lambda x: -9999999 if x == "empty" else x
                    )
                    self.mixed_columns[col] = [-9999999]

            elif col in self.mixed_columns:
                if "empty" in self.df[col].values:
                    self.df[col] = self.df[col].apply(
                        lambda x: -9999999 if x == "empty" else x
                    )
                    self.mixed_columns[col].append(-9999999)

            else:
                if "empty" in self.df[col].values:
                    self.df[col] = self.df[col].apply(
                        lambda x: -9999999 if x == "empty" else x
                    )
                    self.mixed_columns[col] = [-9999999]

        
        # Log transforms
        
        if self.log_columns:
            for log_column in self.log_columns:
                valid_indices = [
                    idx for idx, val in enumerate(self.df[log_column].values)
                    if val != -9999999
                ]

                eps = 1
                lower = np.min(self.df[log_column].iloc[valid_indices].values)
                self.lower_bounds[log_column] = lower

                if lower > 0:
                    self.df[log_column] = self.df[log_column].apply(
                        lambda x: np.log(x) if x != -9999999 else -9999999
                    )
                elif lower == 0:
                    self.df[log_column] = self.df[log_column].apply(
                        lambda x: np.log(x + eps) if x != -9999999 else -9999999
                    )
                else:
                    self.df[log_column] = self.df[log_column].apply(
                        lambda x: np.log(x - lower + eps) if x != -9999999 else -9999999
                    )

        
        for column_index, column in enumerate(self.df.columns):

            # Categorical columns (INDEX-BASED — FIXED)
            if column_index in self.categorical_columns:
                label_encoder = preprocessing.LabelEncoder()
                self.df[column] = self.df[column].astype(str)
                label_encoder.fit(self.df[column])

                self.df[column] = label_encoder.transform(self.df[column])

                self.label_encoder_list.append({
                    "column": column,
                    "label_encoder": label_encoder
                })

                self.column_types["categorical"].append(column_index)

                if column_index in self.general_columns:
                    self.column_types["general"].append(column_index)

                if column_index in self.non_categorical_columns:
                    self.column_types["non_categorical"].append(column_index)

            # Mixed columns (INDEX-BASED — FIXED)
            elif column_index in self.mixed_columns:
                self.column_types["mixed"][column_index] = self.mixed_columns[column_index]

            # General continuous columns
            elif column_index in self.general_columns:
                self.column_types["general"].append(column_index)

        super().__init__()

    
    def inverse_prep(self, data, eps=1):

        df_sample = pd.DataFrame(data, columns=self.df.columns)

        # Inverse label encoding
        for item in self.label_encoder_list:
            le = item["label_encoder"]
            col = item["column"]
            df_sample[col] = df_sample[col].astype(int)
            df_sample[col] = le.inverse_transform(df_sample[col])

        # Inverse log transform
        if self.log_columns:
            for col in self.log_columns:
                lower = self.lower_bounds[col]
                if lower > 0:
                    df_sample[col] = df_sample[col].apply(lambda x: np.exp(x))
                elif lower == 0:
                    df_sample[col] = df_sample[col].apply(
                        lambda x: np.ceil(np.exp(x) - eps)
                        if (np.exp(x) - eps) < 0 else (np.exp(x) - eps)
                    )
                else:
                    df_sample[col] = df_sample[col].apply(
                        lambda x: np.exp(x) - eps + lower
                    )

        
        if self.integer_columns:
            for col in self.integer_columns:
                df_sample[col] = np.round(df_sample[col].values).astype(int)

        df_sample.replace(-9999999, np.nan, inplace=True)
        df_sample.replace('empty', np.nan, inplace=True)

        return df_sample
