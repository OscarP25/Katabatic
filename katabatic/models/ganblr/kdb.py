import numpy as np
from pyitlib import discrete_random_variable as drv


def build_graph(X, y, k=2):
    """
    kDB algorithm (structure learning)
    X and y must be discrete integer-coded arrays
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)

    num_features = X.shape[1]
    x_nodes = list(range(num_features))
    y_node = num_features

    _x = lambda i: X[:, i]
    _x2comb = lambda i, j: (X[:, i], X[:, j])

    sorted_feature_idxs = np.argsort([
        drv.information_mutual(_x(i), y)
        for i in range(num_features)
    ])[::-1]

    edges = []
    for it, target_idx in enumerate(sorted_feature_idxs):
        target_node = x_nodes[target_idx]
        edges.append((y_node, target_node))

        parent_candidate_idxs = sorted_feature_idxs[:it]
        if it <= k:
            for idx in parent_candidate_idxs:
                edges.append((x_nodes[idx], target_node))
        else:
            first_k_parent_mi_idxs = np.argsort([
                drv.information_mutual_conditional(*_x2comb(i, target_idx), y)
                for i in parent_candidate_idxs
            ])[::-1][:k]
            first_k_parent_idxs = parent_candidate_idxs[first_k_parent_mi_idxs]

            for parent_idx in first_k_parent_idxs:
                edges.append((x_nodes[parent_idx], target_node))
    return edges


def get_cross_table(*cols, apply_wt=False):
    """
    Higher dimensional cross-tabulation.

    returns:
      uniq_vals_all_cols: tuple of unique values per col
      xt: ndarray of counts
    """
    if not all(len(col) == len(cols[0]) for col in cols[1:]):
        raise ValueError("all arguments must be same size")
    if len(cols) == 0:
        raise TypeError("get_cross_table() requires at least one argument")

    cols = [np.asarray(c).reshape(-1) for c in cols]

    if apply_wt:
        cols, wt = cols[:-1], cols[-1]
    else:
        wt = 1

    uniq_vals_all_cols, idx = zip(*(np.unique(col, return_inverse=True) for col in cols))
    shape_xt = [uniq_vals_col.size for uniq_vals_col in uniq_vals_all_cols]
    dtype_xt = "float" if apply_wt else "uint"
    xt = np.zeros(shape_xt, dtype=dtype_xt)
    np.add.at(xt, idx, wt)
    return uniq_vals_all_cols, xt


def _get_dependencies_without_y(variables, y_name, kdb_edges):
    """
    evidences of each variable without y.
    """
    dependencies = {}
    kdb_edges_without_y = [edge for edge in kdb_edges if edge[0] != y_name]
    mi_desc_order = {t: i for i, (s, t) in enumerate(kdb_edges) if s == y_name}

    for x in variables:
        current_dependencies = [s for s, t in kdb_edges_without_y if t == x]
        if len(current_dependencies) >= 2:
            sort_dict = {t: mi_desc_order[t] for t in current_dependencies}
            dependencies[x] = sorted(sort_dict)
        else:
            dependencies[x] = current_dependencies
    return dependencies


def _add_uniform(array, noise=1e-5):
    """
    if no count on particular condition for any feature,
    give a uniform prob rather than leave 0
    """
    sum_by_col = np.sum(array, axis=0)
    zero_idxs = (array == 0).astype(int)
    nunique = array.shape[0]

    result = np.zeros_like(array, dtype="float")
    for i in range(array.shape[1]):
        if sum_by_col[i] == 0:
            result[:, i] = array[:, i] + 1.0 / nunique
        elif noise != 0:
            result[:, i] = array[:, i] + noise * zero_idxs[:, i]
        else:
            result[:, i] = array[:, i]
    return result


def get_high_order_feature(X, col, evidence_cols, feature_uniques):
    """
    encode the high order feature of X[col] given evidences X[evidence_cols].
    """
    if evidence_cols is None or len(evidence_cols) == 0:
        return X[:, [col]]
    else:
        base = [1, feature_uniques[col]] + [
            feature_uniques[_col] for _col in evidence_cols[::-1][:-1]
        ]
        cum_base = np.cumprod(base)[::-1]

        cols = evidence_cols + [col]
        high_order_feature = np.sum(X[:, cols] * cum_base, axis=1).reshape(-1, 1)
        return high_order_feature


def get_high_order_constraints(X, col, evidence_cols, feature_uniques):
    """
    find the constraints information for high order feature.
    """
    if evidence_cols is None or len(evidence_cols) == 0:
        unique = feature_uniques[col]
        return np.ones(unique, dtype=bool), np.array([unique])
    else:
        cols = evidence_cols + [col]
        _, cross_table = get_cross_table(*[X[:, i] for i in cols])
        have_value = cross_table != 0

        have_value_reshape = have_value.reshape(-1, have_value.shape[-1])
        high_order_constraints = np.sum(have_value_reshape, axis=-1)

        return have_value, high_order_constraints


class KdbHighOrderFeatureEncoder:
    """
    High order feature encoder that uses the kdb model to retrieve dependencies between features.
    """

    def __init__(self):
        self.dependencies_ = {}
        self.constraints_ = np.array([])
        self.have_value_idxs_ = []
        self.feature_uniques_ = []
        self.high_order_feature_uniques_ = []
        self.edges_ = []
        self.ohe_ = None
        self.k = None

        # stable encoders
        self._ord_ = None

    def fit(self, X, y, k=0):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1)
        self.k = k

        edges = build_graph(X, y, k)
        num_features = X.shape[1]

        if k > 0:
            dependencies = _get_dependencies_without_y(list(range(num_features)), num_features, edges)
        else:
            dependencies = {x: [] for x in range(num_features)}

        self.dependencies_ = dependencies
        self.feature_uniques_ = [len(np.unique(X[:, i])) for i in range(num_features)]
        self.edges_ = edges

        Xk_raw, constraints, have_value_idxs = self.transform(X, return_constraints=True, use_ohe=False)

        from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
        self._ord_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xk_ord = self._ord_.fit_transform(Xk_raw)

        self.ohe_ = OneHotEncoder(handle_unknown="ignore")
        self.ohe_.fit(Xk_ord)

        self.high_order_feature_uniques_ = [len(c) for c in self.ohe_.categories_]
        self.constraints_ = constraints
        self.have_value_idxs_ = have_value_idxs
        return self

    def transform(self, X, return_constraints=False, use_ohe=True):
        X = np.asarray(X)

        Xk = []
        have_value_idxs = []
        constraints = []

        for col, parents in self.dependencies_.items():
            xk = get_high_order_feature(X, col, parents, self.feature_uniques_)
            Xk.append(xk)

            if return_constraints:
                idx, constraint = get_high_order_constraints(X, col, parents, self.feature_uniques_)
                have_value_idxs.append(idx)
                constraints.append(constraint)

        Xk_raw = np.hstack(Xk)

        # If called before fit() finishes
        if self._ord_ is None:
            if return_constraints:
                return Xk_raw, (np.hstack(constraints) if constraints else np.array([])), have_value_idxs
            return Xk_raw

        Xk_ord = self._ord_.transform(Xk_raw)

        if use_ohe:
            X_out = self.ohe_.transform(Xk_ord)
        else:
            X_out = Xk_ord

        if return_constraints:
            concated_constraints = np.hstack(constraints) if constraints else np.array([])
            return X_out, concated_constraints, have_value_idxs
        return X_out
