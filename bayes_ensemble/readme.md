The `bayes_ensemble.py` script contains the code for generating ensembles based on a mixture of distributions.

The `Ensemble` class applies ensemble modeling using a log pooling of distributions. It accepts the `dist` parameter as a string. If `dist = 'log_normal'`, the predictions are parameterized as log-normal distributions, and the ensemble output is also a log-normal distribution. If `dist = 'normal'`, the predictions are parameterized as normal distributions, and the ensemble output follows a normal distribution.

The `Ensemble_linear` class applies the ensemble using a mixture of distributions. In this class, predictions are always parameterized as log-normal distributions.

Both classes are initialized with the `df` parameter, which contains the predictions and must include the following columns: `[`date`, `pred`, `lower`, `upper`, `model_id`]`. Additionally, they require the `order_models` parameter, which is a list specifying the order of models to be used in computing and applying the weights.

Both classes include the `compute_weights` method, which calculates the mixture weights based on the `df_obs` parameter (containing the actual observed data) and the `metric` parameter. The metric parameter can take two values:

`crps`: Minimizes the Continuous Ranked Probability Score (CRPS).
`log_score`: Minimizes the Logarithmic Score.
Additionally, both classes provide the `apply_ensemble` method, which accepts a weights parameter specifying the weights for each model in computing the final ensemble. If no weights are provided, the method defaults to using the weights computed by the `compute_weights` method.