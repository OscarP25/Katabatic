import numpy as np 
import pandas as pd

def bnd_fun(tree, p, forest, feature_names):
    my_tree = forest.estimators_[tree].tree_
    num_nodes = my_tree.node_count
    
    # Initialize bounds
    lb = np.full(shape=(num_nodes, p), fill_value=float('-inf'))
    ub = np.full(shape=(num_nodes, p), fill_value=float('inf'))
    
    # Traverse tree
    for i in range(num_nodes):
        left_child = my_tree.children_left[i]
        right_child = my_tree.children_right[i]
        
        if left_child > -1: # Not a leaf
            ub[left_child, :] = ub[right_child, :] = ub[i, :]
            lb[left_child, :] = lb[right_child, :] = lb[i, :]
            
            if left_child != right_child:
                ub[left_child, my_tree.feature[i]] = my_tree.threshold[i]
                lb[right_child, my_tree.feature[i]] = my_tree.threshold[i]

    leaves = np.nonzero(my_tree.children_left < 0)[0]
    n_leaves = len(leaves)
    
    # --- TROUBLESHOOTING FIX: Force flat lists ---
    # numpy arrays of shape (N,) and (N,1) can cause "length" errors in loose dicts.
    # .flatten().tolist() ensures we have simple 1D python lists.
    
    l_data = {
        'tree': np.full(n_leaves, tree).flatten().tolist(), 
        'leaf': leaves.flatten().tolist()
    }
    u_data = {
        'tree': np.full(n_leaves, tree).flatten().tolist(), 
        'leaf': leaves.flatten().tolist()
    }
    
    for k, col in enumerate(feature_names):
        # Extract, flatten, and convert to list
        l_data[col] = lb[leaves, k].flatten().tolist()
        u_data[col] = ub[leaves, k].flatten().tolist()
        
    try:
        l = pd.DataFrame(l_data)
        u = pd.DataFrame(u_data)
    except ValueError as e:
        print(f"DEBUG: Error in bnd_fun for tree {tree}")
        for k, v in l_data.items():
            print(f"Col: {k}, Length: {len(v)}")
        raise e
    
    ret = pd.merge(
        left=pd.melt(l, id_vars=['tree', 'leaf'], value_name='min'),
        right=pd.melt(u, id_vars=['tree', 'leaf'], value_name='max'),
        on=['tree','leaf', 'variable']
    )
    return ret