class GeneratorPredictor:
    def train_gen_pred(self, loader, *args, **kwargs):
        """
        Running training for generation task and predictor task.

        Args:
            loader: The data loader for loading training samples.
        """

        raise NotImplementedError(
            "The function train_gen_pred is not implemented!"
        )

    def run_rand_gen(self, *args, **kwargs):
        """
        Running graph generation for random generation task.
        """

        raise NotImplementedError(
            "The function run_rand_gen is not implemented!"
        )

    def run_prop_opt(self, *args, **kwargs):
        """
        Running graph generation for property optimization task.
        """

        raise NotImplementedError(
            "The function run_prop_opt is not implemented!"
        )
