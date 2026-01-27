import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._forest import _generate_unsampled_indices
import scipy
from . import utils

class arf:
  """Implements Adversarial Random Forests (ARF)"""   
  def __init__(self, x,  num_trees = 30, delta = 0,  max_iters =10, early_stop = True, verbose = False, min_node_size = 5, **kwargs):

    # assertions
    assert isinstance(x, pd.core.frame.DataFrame), "expected pandas DataFrame"
    assert len(set(list(x))) == x.shape[1], "duplicate column names"
    
    # initialize values (Reset index to prevent alignment errors)
    x_real = x.copy().reset_index(drop=True)
    self.p = x_real.shape[1]
    self.orig_colnames = list(x_real)
    self.num_trees = num_trees

    # Data Type Conversions
    self.object_cols = x_real.dtypes == "object"
    for col in list(x_real):
      if self.object_cols[col]:
        x_real[col] = x_real[col].astype('category')
    
    self.factor_cols = x_real.dtypes == "category"
    self.levels = {}
    for col in list(x_real):
      if self.factor_cols[col]:
        self.levels[col] = x_real[col].cat.categories
        x_real[col] = x_real[col].cat.codes
    
    # Initial Synthetic Data
    x_synth_dict = {col: x_real[col].sample(frac=1, replace=False).values.flatten() for col in x_real.columns}
    x_synth = pd.DataFrame(x_synth_dict, columns=x_real.columns)
    
    x = pd.concat([x_real, x_synth], axis=0, ignore_index=True)
    y = np.concatenate([np.zeros(x_real.shape[0]), np.ones(x_real.shape[0])])

    self.x_real = x_real
    clf_0 = RandomForestClassifier(oob_score=True, n_estimators=self.num_trees, min_samples_leaf=min_node_size, **kwargs) 
    clf_0.fit(x, y)

    iters = 0
    acc_0 = clf_0.oob_score_
    acc = [acc_0]

    if verbose: print(f'Initial accuracy: {acc_0}')

    if (acc_0 > 0.5 + delta and iters < max_iters):
      converged = False
      while (not converged):
        # Adversarial Loop
        nodeIDs = clf_0.apply(self.x_real)
        
        # Prepare matching dataframe
        x_real_obs = x_real.copy()
        x_real_obs['obs'] = range(len(x_real))
        
        nodeIDs_pd = pd.DataFrame(nodeIDs)
        nodeIDs_pd['obs'] = range(len(x_real))
        tmp = nodeIDs_pd.melt(id_vars=['obs'], value_name="leaf", var_name="tree")

        x_real_obs = pd.merge(left=x_real_obs, right=tmp, on=['obs'], sort=False)
        x_real_obs.drop('obs', axis=1, inplace=True)

        tmp.drop("obs", axis=1, inplace=True)
        tmp = tmp.sample(x_real.shape[0], axis=0, replace=True)
        tmp = pd.Series(tmp.value_counts(sort=False), name='cnt').reset_index()
        draw_from = pd.merge(left=tmp, right=x_real_obs, on=['tree', 'leaf'], sort=False)

        # Synthetic generation step
        grpd = draw_from.groupby(['tree', 'leaf'])
        x_synth_dfs = []
        
        for ind in grpd.groups.keys():
            group = grpd.get_group(ind)
            count = int(group['cnt'].iloc[0])
            feature_cols = [c for c in group.columns if c not in ['cnt', 'tree', 'leaf', 'obs']]
            
            # Marginal sampling within leaf
            sampled_data = {col: group[col].sample(n=count, replace=True).values.flatten() for col in feature_cols}
            x_synth_dfs.append(pd.DataFrame(sampled_data))
            
        if x_synth_dfs:
            x_synth = pd.concat(x_synth_dfs, axis=0, ignore_index=True)
            x_synth = x_synth[x_real.columns] # Enforce order
        else:
            x_synth = pd.DataFrame(columns=x_real.columns)

        del(nodeIDs, nodeIDs_pd, tmp, x_real_obs, draw_from)

        x = pd.concat([x_real, x_synth], axis=0, ignore_index=True)
        y = np.concatenate([np.zeros(x_real.shape[0]), np.ones(x_synth.shape[0])])
        
        clf_1 = RandomForestClassifier(oob_score=True, n_estimators=self.num_trees, min_samples_leaf=min_node_size, **kwargs) 
        clf_1.fit(x, y)

        acc_1 = clf_1.oob_score_
        acc.append(acc_1)
        iters += 1
        
        plateau = True if early_stop and acc[iters] > acc[iters - 1] else False
        if verbose:
          print(f"Iter {iters}: acc {acc_1}")
        
        if (acc_1 <= 0.5 + delta or iters >= max_iters or plateau):
          converged = True
        else:
          clf_0 = clf_1
          
    self.clf = clf_0
    self.acc = acc 
        
    # Pruning
    pred = self.clf.apply(self.x_real)
    for tree_num in range(0, self.num_trees):
      tree = self.clf.estimators_[tree_num]
      left = tree.tree_.children_left
      right = tree.tree_.children_right
      leaves = np.where(left < 0)[0]

      unique, counts = np.unique(pred[:, tree_num], return_counts=True)
      to_prune = unique[counts < min_node_size]
      to_prune = np.concatenate([to_prune, np.setdiff1d(leaves, unique)])

      while len(to_prune) > 0:
        for tp in to_prune:
          parent = np.where(left == tp)[0]
          if len(parent) > 0:
            left[parent] = right[parent]
          else:
            parent = np.where(right == tp)[0]
            right[parent] = left[parent]
        to_prune = np.where(np.isin(left, to_prune))[0]

  def forde(self, dist="truncnorm", oob=False, alpha=0):
    self.dist = dist
    self.oob = oob
    self.alpha = alpha

    pred = self.clf.apply(self.x_real)
    
    if self.oob:
      for tree in range(self.num_trees):
        idx_oob = np.isin(range(self.x_real.shape[0]), _generate_unsampled_indices(self.clf.estimators_[tree].random_state, self.x.shape[0], self.x.shape[0]))
        pred[np.invert(idx_oob), tree] = -1
    
    # Calculate Bounds
    bnds = pd.concat([utils.bnd_fun(tree=j, p=self.p, forest=self.clf, feature_names=self.orig_colnames) for j in range(self.num_trees)])
    bnds['f_idx'] = bnds.groupby(['tree', 'leaf']).ngroup()

    # Calculate Coverage
    bnds_list = []
    for t in range(self.num_trees):
      unique, freq = np.unique(pred[:,t], return_counts=True)
      vv = pd.DataFrame({'leaf': unique, 'cvg': freq/pred.shape[0]})
      zz = bnds[bnds['tree'] == t]
      bnds_list.append(pd.merge(left=vv, right=zz, on=['leaf']))
    
    bnds = pd.concat(bnds_list, axis=0, ignore_index=True)

    if np.invert(self.factor_cols).any():
      bnds.loc[bnds['cvg'] == 1/pred.shape[0], 'cvg'] = 0
    
    bnds = bnds[bnds['cvg'] > 0]
    bnds.rename(columns={'leaf': 'nodeid'}, inplace=True)
    self.bnds = bnds

    # Params
    self.params = pd.DataFrame()
    if np.invert(self.factor_cols).any():
      for tree in range(self.num_trees):
        dt = self.x_real.loc[:, np.invert(self.factor_cols)].copy()
        dt["tree"] = tree
        dt["nodeid"] = pred[:,tree]
        long = pd.merge(right=bnds[['tree', 'nodeid','variable', 'min', 'max', 'f_idx']], left=pd.melt(dt[dt["nodeid"] >= 0], id_vars=["tree", "nodeid"]), on=['tree', 'nodeid', 'variable'], how='left')
        
        if self.dist == "truncnorm":
          res = long.groupby(['tree',"nodeid", "variable"], as_index=False).agg(mean=("value", "mean"), sd=("value", "std"), min=("min", "min"), max=("max", "max"))
        else:
          raise ValueError('unknown distribution')
        self.params = pd.concat([self.params, res])
    
    # Probs
    self.class_probs = pd.DataFrame()
    if self.factor_cols.any():
      for tree in range(self.num_trees):
        dt = self.x_real.loc[:, self.factor_cols].copy()
        dt["tree"] = tree
        dt["nodeid"] = pred[:,tree]
        dt = pd.melt(dt[dt["nodeid"] >= 0], id_vars=["tree", "nodeid"])
        long = pd.merge(left=dt, right=bnds, on=['tree','nodeid', 'variable'])
        long['count_var'] = long.groupby(['tree', 'nodeid', 'variable'])['variable'].transform('count')
        long['count_var_val'] = long.groupby(['tree', 'nodeid', 'variable', 'value'])['variable'].transform('count')
        long.drop_duplicates(inplace=True)
        if self.alpha == 0:
          long['prob'] = long['count_var_val'] / long['count_var'] 
        else:
          long['k'] = long.groupby(['variable'])['value'].transform('nunique')  
          long.loc[long['min'] == float('-inf') , 'min'] = 0.5 - 1
          long.loc[long['max'] == float('inf') , 'max'] = long['k'] + 0.5 - 1
          
          # Boolean masking
          min_mod = np.round(long['min'] % 1, 2)
          max_mod = np.round(long['max'] % 1, 2)
          
          mask_min = ~np.isclose(min_mod, 0.5)
          long.loc[mask_min, 'min'] = long.loc[mask_min, 'min'] - 0.5
          
          mask_max = ~np.isclose(max_mod, 0.5)
          long.loc[mask_max, 'max'] = long.loc[mask_max, 'max'] + 0.5
          
          long['k'] = long['max'] - long['min']  
          
          tmp = long[['f_idx','tree', "nodeid", 'variable', 'min','max']].copy()
          tmp['rep_min'] = tmp['min'] + 0.5 
          tmp['rep_max'] = tmp['max'] - 0.5 
          tmp['levels'] = tmp.apply(lambda row: list(range(int(row['rep_min']), int(row['rep_max'] + 1))), axis=1)
          tmp = tmp.explode('levels')
          
          # Concatenation for variable-length categories
          cat_val_list = [pd.DataFrame({'variable': col, 'value': self.levels[col]}) for col in self.levels]
          cat_val = pd.concat(cat_val_list, ignore_index=True)
          cat_val['levels'] = cat_val['value']

          tmp = pd.merge(left=tmp, right=cat_val, on=['variable', 'levels'])[['variable', 'f_idx','tree', "nodeid",'value']]
          tmp = pd.merge(left=tmp, right=long[['f_idx', 'variable', 'tree', "nodeid",'count_var', 'k']], on=['f_idx', "nodeid", 'variable', 'tree'])
          long = pd.merge(left=tmp, right=long, on=['f_idx','tree',"nodeid", 'variable','value','count_var','k'], how='left')
          long.loc[long['count_var_val'].isna(), 'count_var_val'] = 0
          long = long[['f_idx','tree',"nodeid", 'variable', 'value', 'count_var_val', 'count_var', 'k']].drop_duplicates()
          long['prob'] = (long['count_var_val'] + self.alpha) / (long['count_var'] + self.alpha*long['k'])
          long['value'] = long['value'].astype('int8')
        
        long = long[['f_idx','tree', "nodeid", 'variable', 'value','prob']]
        self.class_probs = pd.concat([self.class_probs, long])
        
    return {"cnt": self.params, "cat": self.class_probs, 
            "forest": self.clf, "meta": pd.DataFrame(data={"variable": self.orig_colnames, "family": self.dist})}

  def forge(self, n):
    try:
      getattr(self, 'bnds')
    except AttributeError:
      raise AttributeError('need density estimates -> run .forde() first!')

    unique_bnds = self.bnds[['tree', 'nodeid', 'cvg']].drop_duplicates()
    
    draws = np.random.choice(a=range(unique_bnds.shape[0]), p=unique_bnds['cvg'] / self.num_trees, size=n)
    
    sampled_trees_nodes = unique_bnds[['tree','nodeid']].iloc[draws,].reset_index(drop=True).reset_index().rename(columns={'index': 'obs'})

    if np.invert(self.factor_cols).any():
      obs_params = pd.merge(sampled_trees_nodes, self.params, on=["tree", "nodeid"]).sort_values(by=['obs'], ignore_index=True)
    
    if self.factor_cols.any():
      obs_probs = pd.merge(sampled_trees_nodes, self.class_probs, on=["tree", "nodeid"]).sort_values(by=['obs'], ignore_index=True)
    
    data_new = pd.DataFrame(index=range(n), columns=range(self.p))
    for j in range(self.p): 
      colname = self.orig_colnames[j]
      
      if self.factor_cols.iloc[j]:
        data_new.isetitem(j, obs_probs[obs_probs["variable"] == colname].groupby("obs").sample(weights="prob")["value"].reset_index(drop=True))
      else:
        if self.dist == "truncnorm":
         myclip_a = obs_params.loc[obs_params["variable"] == colname, "min"]
         myclip_b = obs_params.loc[obs_params["variable"] == colname, "max"]
         myloc = obs_params.loc[obs_params["variable"] == colname, "mean"]
         myscale = obs_params.loc[obs_params["variable"] == colname, "sd"]
         data_new.isetitem(j, scipy.stats.truncnorm(a=(myclip_a - myloc) / myscale, b=(myclip_b - myloc) / myscale, loc=myloc, scale=myscale).rvs(size=n))
        else:
          raise ValueError('Other distributions not yet implemented')
    
    data_new = data_new.set_axis(self.orig_colnames, axis=1, copy=False)
    
    # Convert categories back to category   
    for col in self.orig_colnames:
      if self.factor_cols[col]:
        codes = data_new[col].fillna(-1).astype(int)
        data_new[col] = pd.Categorical.from_codes(codes, categories=self.levels[col])

    # Convert object columns back to object
    for col in self.orig_colnames:
      if self.object_cols[col]:
        data_new[col] = data_new[col].astype("object")

    return data_new