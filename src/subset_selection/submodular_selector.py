from .subset_selector import SubsetSelector
from encoders import BaseEncoder
from typing import Dict
import submodlib
from src.utils.dist_utils.get_model_dependent_utility_kernel import ModelDependentICLUtility
from src.utils.dist_utils.get_model_independent_kernel import ModelIndependentUtility

# Path: subset_selectors/submodular_selector.py
# Contains implementation of SubsetSelector class that uses submodular function to select the subset of samples.
# The encoder class should be a subclass of BaseEncoder class defined in encoders/base_encoder.py

submod_function_dict = {
    "fl": submodlib.FacilityLocationFunction,
    "logdet": submodlib.LogDeterminantFunction,
    "gc": submodlib.GraphCutFunction,
    "disp_min": submodlib.DisparityMinFunction,
    "disp_sum": submodlib.DisparitySumFunction,}

default_submodular_fn_params_dict = {
    "fl": {"mode": "dense", "separate_rep": False},
    "logdet": {"mode": "dense", "lambdaVal": 1},
    "gc": {"mode": "dense", "separate_rep": False, "lambdaVal": 0.4},
    "disp_min": {"mode": "dense"},
    "disp_sum": {"mode": "dense"},}


class SubmodularSelector(SubsetSelector):
    def __init__(self, 
                 encoder: BaseEncoder, 
                 device: str, 
                 similarity_fn: str,
                 submodular_fn: str, 
                 submodular_fn_params: Dict = None,
                 model_dependent: bool = False,
                 similarity_fn_params: Dict = None):
        super().__init__(encoder, device)
        self.submodular_fn = submodular_fn
        self.submodular_fn_params = submodular_fn_params
        self.similarity_fn = similarity_fn
        self.similarity_fn_params = similarity_fn_params
        if model_dependent:
            self.utility = ModelDependentICLUtility(model=self.similarity_fn_params['model'], 
                                                    tokenizer=self.similarity_fn_params['tokenizer'],
                                                    device=self.device)
        else:
            self.utility = ModelIndependentUtility()


    def select_subset(self, 
                      data, 
                      subset_size: int):
        # Encode the data samples
        data_repr = self.encoder.encode(data)
        if mode
        # Initialize the similarity function
        similarity_fn = similarity_function_dict[self.similarity_fn]
        # Compute the similarity matrix
        sim_mat = similarity_fn(data_repr)
        # Initialize the submodular function
        if self.submodular_fn_params is None:
            submod_fn_params = default_submodular_fn_params_dict[self.submodular_fn]
        else:
            submod_fn_params = self.submodular_fn_params

        if self.submodular_fn == "gc":
            submod_fn_params['ggsijs'] = sim_mat
        else:
            submod_fn_params['sijs'] = sim_mat

        submod_fn_params['n'] = data_repr.shape[0]

        submod_fn = submod_function_dict[self.submodular_fn](**submod_fn_params)
        # Select the subset of samples
        if self.submodular_fn == "disp_min":
            optim = "NaiveGreedy"
        else:
            optim = "LazyGreedy"
        assert subset_size <= data_repr.shape[0], "Subset size should be less than the total number of samples"
        subset = submod_fn.maximize(budget=subset_size,
                                    optimizer=optim,
                                    stopIfZeroGain=False,
                                    stopIfNegativeGain=False,
                                    verbose=False)
        return subset

                                                                     