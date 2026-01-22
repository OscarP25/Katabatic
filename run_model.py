def main():
    import importlib
    import sys

    model_map = {
        "codi": "CODI",
        "ctgan": "CTGANModel",
        "ganblr": "GANBLR",
        "great": "GReaT",
        "medgan": "MEDGAN",
        "pategan": "PATEGAN",
        "tabddpm": "Tabddpm",
        "tabsyn": "TabSyn",
        "tabfairgdt": "TabFairGDT",
        "tabfairgan": "TFG",
        "arf": "KatabaticARF"
    }

    if len(sys.argv) < 3:
        print("Usage: python run_model.py <model_name> <dataset>")
        sys.exit(1)

    model_name = sys.argv[1].lower()
    dataset_name = sys.argv[2]

    if model_name not in model_map:
        print(f"Error: Model '{model_name}' not found in map.")
        sys.exit(1)

    if model_name == "arf":
        module_name = f"katabatic.models.{model_name}.adapter"
    else:
        module_name = f"katabatic.models.{model_name}"
        
    class_name = model_map[model_name]

    try:
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
    except (ModuleNotFoundError, AttributeError) as e:
        print(f"Error loading model '{model_name}' ({class_name}) from {module_name}: {e}")
        sys.exit(1)

    from katabatic.pipeline.train_test_split.pipeline import \
        TrainTestSplitPipeline
    from utils import discretize_preprocess

    raw_path = f"raw_data/{dataset_name}.csv"
    discretized_path = f"discretized_data/{dataset_name}.csv"
    output_dir = f"sample_data/{dataset_name}"
    synthetic_dir = f"synthetic/{dataset_name}/{model_name}"

    model_config = {}

    if model_name == "tabddpm":
        model_config = {
            "steps": 15000,
            "num_timesteps": 1000,
            "batch_size": 256,
            "d_layers": (256, 256, 256, 256),
            "use_ema": True,
            "lr": 0.002,
            "scheduler": "cosine"
        }

    elif model_name == "arf":
        print(f"\n--- Configuration for {model_name.upper()} ---")
        model_config = {
            "num_trees": 50,      # Number of trees in the forest
            "max_iters": 10,      # Adversarial iterations
            "min_node_size": 5,   # Minimum samples per leaf
            "delta": 0.0          # Convergence tolerance
        }

    elif model_name == "tabfairgdt":
        print(f"\n--- Configuration for {model_name.upper()} ---")
        protected_col = input("Protected Attribute (S): ").strip()
        model_config = {
            "protected_attribute": protected_col,
            "lambda_val": 1
        }

    elif model_name == "tabfairgan":
        print(f"\n--- Configuration for {model_name.upper()} ---")
        s_col = input("Protected Attribute (S) [e.g. sex]: ").strip()
        y_col = input("Target Attribute (Y) [e.g. income]: ").strip()
        s_under = input("Underprivileged Group Value (S_under) [e.g. ' Female']: ").strip()
        y_desire = input("Desired Outcome Value (Y_desire) [e.g. ' >50K']: ").strip()

        model_config = {
            "epochs": 200,
            "batch_size": 256,
            "fairness_config": {
                "fair_epochs": 50,
                "lamda": 0.5,
                "S": s_col,
                "Y": y_col,
                "S_under": s_under,
                "Y_desire": y_desire
            }
        }

    try:
        discretize_preprocess(
            file_path=raw_path,
            output_path=discretized_path,
            bins=10,
            strategy='uniform'
        )

        pipeline = TrainTestSplitPipeline(model=model_class)

    except Exception as e:
        print(f"Error preparing pipeline: {e}")
        sys.exit(1)

    print(f"Starting pipeline for {model_name} on {dataset_name}")
    
    pipeline.run(
        input_csv=discretized_path,
        output_dir=output_dir,
        synthetic_dir=synthetic_dir,
        real_test_dir=output_dir,
        **model_config
    )

if __name__ == "__main__":
    main()