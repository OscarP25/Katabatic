class Evaluation:
    """
    Base class for evaluation.
    """

    def __init__(self, model=None, dataset=None, **kwargs):
        self.model = model
        self.dataset = dataset
        self.kwargs = kwargs
        self.results_ = None

    def evaluate(self):
        """
        Evaluate the model on the dataset.
        """
        raise NotImplementedError("Subclasses should implement this method.")
