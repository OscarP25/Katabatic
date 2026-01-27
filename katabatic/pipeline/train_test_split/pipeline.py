from katabatic.pipeline.base_pipeline import Pipeline
from katabatic.models.base_model import Model
from katabatic.evaluate.tstr.evaluation import TSTREvaluation
from katabatic.utils.split_dataset import split_dataset


class TrainTestSplitPipeline(Pipeline):
    """
    Version 1 of the pipeline.
    """

    _evaluations = [TSTREvaluation]

    def __init__(self, model: Model, evaluations=None, override_evaluations=False):
        super().__init__(model)

        if evaluations and override_evaluations:
            self._evaluations = evaluations
        elif evaluations:
            self._evaluations.extend(evaluations)

    def run(self, *args, **kwargs):
        """
        Run the train-test-split pipeline with the given arguments.

        Expected kwargs:
          - input_csv   : path to discretized CSV (e.g. discretized_data/car.csv)
          - output_dir  : directory for real train/test (e.g. sample_data/car)
          - synthetic_dir (optional): where to store synthetic data
        """
        current_model = self.model()

        # -------- required arguments ----------
        input_csv = kwargs.pop("input_csv", None)
        output_dir = kwargs.pop("output_dir", None)

        if not input_csv or not output_dir:
            raise ValueError("Both 'input_csv' and 'output_dir' must be provided.")

        # synthetic_dir: where PATE-GAN will write x_synth/y_synth
        synthetic_dir = kwargs.get("synthetic_dir", output_dir)

        # -------- 1) split real data ----------
        # split_dataset does NOT know about synthetic_dir, so remove it
        split_kwargs = dict(kwargs)
        split_kwargs.pop("synthetic_dir", None)

        split_dataset(input_csv, output_dir, *args, **split_kwargs)

        # -------- 2) train model on real train ----------
        # current_model (PATEGAN) *does* use synthetic_dir from kwargs
        current_model.train(output_dir, *args, **kwargs)

        # -------- 3) run TSTR evaluation ----------
        eval_kwargs = dict(kwargs)
        # avoid passing configs evaluation doesn’t expect
        eval_kwargs.pop("config", None)

        # These two are REQUIRED by TSTREvaluation
        eval_kwargs["synthetic_dir"] = synthetic_dir
        # real_test_dir is where x_test/y_test live
        eval_kwargs["real_test_dir"] = output_dir

        results = {}
        for evaluation in self._evaluations:
            eval_instance = evaluation(*args, **eval_kwargs)
            eval_result = eval_instance.evaluate()
            # collect results if evaluate() returns anything
            if eval_result is not None:
                results[evaluation.__name__] = eval_result

        # Optional: return all evaluation results
        return results

    def __repr__(self):
        return f"TrainTestSplitPipeline(name={self.model})"
