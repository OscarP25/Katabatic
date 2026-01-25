import numpy as np
import pandas as pd
import torch
from sklearn.mixture import BayesianGaussianMixture


# DATA TRANSFORMER

class DataTransformer:

    def __init__(
        self,
        train_data=pd.DataFrame,
        categorical_list=[],
        mixed_dict={},
        general_list=[],
        non_categorical_list=[],
        n_clusters=10,
        eps=0.005
    ):
        self.meta = None
        self.n_clusters = n_clusters
        self.eps = eps
        self.train_data = train_data
        self.categorical_columns = categorical_list
        self.mixed_columns = mixed_dict
        self.general_columns = general_list
        self.non_categorical_columns = non_categorical_list

    
    # Metadata
    
    def get_metadata(self):
        meta = []

        for index in range(self.train_data.shape[1]):
            column = self.train_data.iloc[:, index]

            if index in self.categorical_columns:
                if index in self.non_categorical_columns:
                    meta.append({
                        "name": index,
                        "type": "continuous",
                        "min": column.min(),
                        "max": column.max(),
                    })
                else:
                    mapper = column.value_counts().index.tolist()

                  
                    if len(mapper) == 0:
                        mapper = [0]

                    meta.append({
                        "name": index,
                        "type": "categorical",
                        "size": len(mapper),
                        "i2s": mapper
                    })

            elif index in self.mixed_columns:
                meta.append({
                    "name": index,
                    "type": "mixed",
                    "min": column.min(),
                    "max": column.max(),
                    "modal": self.mixed_columns[index]
                })
            else:
                meta.append({
                    "name": index,
                    "type": "continuous",
                    "min": column.min(),
                    "max": column.max(),
                })

        return meta

   
    def fit(self):
        data = self.train_data.values
        self.meta = self.get_metadata()

        self.model = []
        self.ordering = []
        self.output_info = []
        self.output_dim = 0
        self.components = []
        self.filter_arr = []

        for id_, info in enumerate(self.meta):

            
            if info["type"] == "continuous":

                if id_ not in self.general_columns:
                    gm = BayesianGaussianMixture(
                        n_components=self.n_clusters,
                        weight_concentration_prior_type="dirichlet_process",
                        weight_concentration_prior=0.001,
                        max_iter=100,
                        n_init=1,
                        random_state=42,
                    )
                    gm.fit(data[:, id_].reshape(-1, 1))

                    mode_freq = (
                        pd.Series(gm.predict(data[:, id_].reshape(-1, 1)))
                        .value_counts()
                        .keys()
                    )

                    old_comp = gm.weights_ > self.eps
                    comp = [
                        True if (i in mode_freq) and old_comp[i] else False
                        for i in range(self.n_clusters)
                    ]

                    
                    if not any(comp):
                        comp[np.argmax(gm.weights_)] = True

                    self.model.append(gm)
                    self.components.append(comp)
                    self.output_info += [(1, "tanh", "no_g"), (np.sum(comp), "softmax")]
                    self.output_dim += 1 + np.sum(comp)

                else:
                    self.model.append(None)
                    self.components.append(None)
                    self.output_info += [(1, "tanh", "yes_g")]
                    self.output_dim += 1

          
            elif info["type"] == "mixed":

                gm1 = BayesianGaussianMixture(
                    n_components=self.n_clusters,
                    weight_concentration_prior_type="dirichlet_process",
                    weight_concentration_prior=0.001,
                    max_iter=100,
                    n_init=1,
                    random_state=42,
                )

                gm2 = BayesianGaussianMixture(
                    n_components=self.n_clusters,
                    weight_concentration_prior_type="dirichlet_process",
                    weight_concentration_prior=0.001,
                    max_iter=100,
                    n_init=1,
                    random_state=42,
                )

                gm1.fit(data[:, id_].reshape(-1, 1))

                filter_arr = [v not in info["modal"] for v in data[:, id_]]
                self.filter_arr.append(filter_arr)

                gm2.fit(data[:, id_][filter_arr].reshape(-1, 1))

                mode_freq = (
                    pd.Series(
                        gm2.predict(data[:, id_][filter_arr].reshape(-1, 1))
                    )
                    .value_counts()
                    .keys()
                )

                old_comp = gm2.weights_ > self.eps
                comp = [
                    True if (i in mode_freq) and old_comp[i] else False
                    for i in range(self.n_clusters)
                ]

                
                if not any(comp):
                    comp[np.argmax(gm2.weights_)] = True

                self.model.append((gm1, gm2))
                self.components.append(comp)

                self.output_info += [
                    (1, "tanh", "no_g"),
                    (np.sum(comp) + len(info["modal"]), "softmax"),
                ]
                self.output_dim += 1 + np.sum(comp) + len(info["modal"])

            
            else:
                self.model.append(None)
                self.components.append(None)
                self.output_info += [(max(info["size"], 1), "softmax")]
                self.output_dim += max(info["size"], 1)

    
    def transform(self, data):
        values = []
        mixed_counter = 0

        for id_, info in enumerate(self.meta):
            current = data[:, id_]

            if info["type"] == "continuous":
                if id_ not in self.general_columns:
                    current = current.reshape(-1, 1)
                    means = self.model[id_].means_.reshape(1, -1)
                    stds = np.sqrt(self.model[id_].covariances_).reshape(1, -1)

                    features = (current - means) / (4 * stds)
                    probs = self.model[id_].predict_proba(current)

                    features = features[:, self.components[id_]]
                    probs = probs[:, self.components[id_]]

                    opt_sel = np.array([
                        np.random.choice(len(p), p=(p + 1e-6) / np.sum(p + 1e-6))
                        for p in probs
                    ])

                    features = features[np.arange(len(features)), opt_sel].reshape(-1, 1)
                    features = np.clip(features, -0.99, 0.99)

                    onehot = np.zeros_like(probs)
                    onehot[np.arange(len(probs)), opt_sel] = 1

                    col_sum = onehot.sum(axis=0)
                    order = np.argsort(-col_sum)
                    self.ordering.append(order)

                    values += [features, onehot[:, order]]

                else:
                    self.ordering.append(None)
                    current = (current - info["min"]) / (info["max"] - info["min"])
                    current = current * 2 - 1
                    values.append(current.reshape(-1, 1))

            elif info["type"] == "mixed":
                current = current.reshape(-1, 1)
                filter_arr = self.filter_arr[mixed_counter]
                current_f = current[filter_arr]

                means = self.model[id_][1].means_.reshape(1, -1)
                stds = np.sqrt(self.model[id_][1].covariances_).reshape(1, -1)

                features = (current_f - means) / (4 * stds)
                probs = self.model[id_][1].predict_proba(current_f)

                features = features[:, self.components[id_]]
                probs = probs[:, self.components[id_]]

                opt_sel = np.array([
                    np.random.choice(len(p), p=(p + 1e-6) / np.sum(p + 1e-6))
                    for p in probs
                ])

                features = features[np.arange(len(features)), opt_sel].reshape(-1, 1)
                features = np.clip(features, -0.99, 0.99)

                onehot = np.zeros_like(probs)
                onehot[np.arange(len(probs)), opt_sel] = 1

                final = np.zeros((len(data), 1 + onehot.shape[1]))
                idx_f = 0

                for i in range(len(data)):
                    if not filter_arr[i]:
                        final[i, 1] = 1
                    else:
                        final[i, 0] = features[idx_f]
                        final[i, 1:] = onehot[idx_f]
                        idx_f += 1

                col_sum = final[:, 1:].sum(axis=0)
                order = np.argsort(-col_sum)
                self.ordering.append(order)
                final[:, 1:] = final[:, 1:][:, order]

                values.append(final)
                mixed_counter += 1

            else:
                self.ordering.append(None)

                if info["size"] <= 1:
                    col = np.ones((len(data), 1))
                else:
                    col = np.zeros((len(data), info["size"]))
                    idx = list(map(info["i2s"].index, current))
                    col[np.arange(len(data)), idx] = 1

                values.append(col)

        return np.concatenate(values, axis=1)

    
    def inverse_transform(self, data):
        data_t = np.zeros((len(data), len(self.meta)))
        st = 0

        for id_, info in enumerate(self.meta):

            if info["type"] == "continuous":
                if id_ not in self.general_columns:
                    u = data[:, st]
                    v = data[:, st + 1: st + 1 + np.sum(self.components[id_])]
                    order = self.ordering[id_]
                    v = v[:, np.argsort(order)]

                    means = self.model[id_].means_.reshape(-1)
                    stds = np.sqrt(self.model[id_].covariances_).reshape(-1)
                    p = np.argmax(v, axis=1)
                    data_t[:, id_] = u * 4 * stds[p] + means[p]
                    st += 1 + np.sum(self.components[id_])
                else:
                    u = (data[:, st] + 1) / 2
                    data_t[:, id_] = u * (info["max"] - info["min"]) + info["min"]
                    st += 1

            elif info["type"] == "mixed":
                u = data[:, st]
                v = data[:, st + 1: st + 1 + np.sum(self.components[id_])]
                order = self.ordering[id_]
                v = v[:, np.argsort(order)]

                means = self.model[id_][1].means_.reshape(-1)
                stds = np.sqrt(self.model[id_][1].covariances_).reshape(-1)
                p = np.argmax(v, axis=1)

                data_t[:, id_] = u * 4 * stds[p] + means[p]
                st += 1 + np.sum(self.components[id_]) + len(info["modal"])

            elif info["type"] == "categorical":
                col = data[:, st: st + info["size"]]
                idx = np.argmax(col, axis=1)
                data_t[:, id_] = [info["i2s"][i] for i in idx]
                st += info["size"]

        return data_t, 0


class ImageTransformer:

    def __init__(self, side):
        self.height = side

    def transform(self, data):
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)

        batch = data.size(0)
        target_dim = self.height * self.height
        current_dim = data.size(1)

        if current_dim < target_dim:
            padding = torch.zeros(batch, target_dim - current_dim, device=data.device)
            data = torch.cat([data, padding], dim=1)
        elif current_dim > target_dim:
            data = data[:, :target_dim]

        return data.view(batch, 1, self.height, self.height)

    def inverse_transform(self, data):
        if data.dim() == 4:
            data = data.view(data.size(0), -1)

        batch = data.size(0)
        target_dim = self.height * self.height
        current_dim = data.size(1)

        if current_dim < target_dim:
            pad = torch.zeros(batch, target_dim - current_dim, device=data.device)
            data = torch.cat([data, pad], dim=1)
        elif current_dim > target_dim:
            data = data[:, :target_dim]

        return data
