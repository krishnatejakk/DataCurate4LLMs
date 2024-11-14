from .subset_selector import SubsetSelector
from encoders import BaseEncoder
from typing import Dict
import submodlib
from utils import *

# Path: subset_selectors/submodular_selector.py
# Contains implementation of SubsetSelector class that uses submodular function to select the subset of samples.
# The encoder class should be a subclass of BaseEncoder class defined in encoders/base_encoder.py

submod_function_dict = {
    "flmi": submodlib.FacilityLocationMutualInformationFunction,
    "flvmi": submodlib.FacilityLocationVariantMutualInformationFunction,
    "gcmi": submodlib.GraphCutMutualInformationFunction,
    "logdetmi": submodlib.LogDeterminantMutualInformationFunction,}

similarity_function_dict = {
    "cosine_np": compute_scaled_cosine_similarity_np,
    "dot_np": compute_scaled_dot_product_kernel_np,
    "rbf_np": compute_rbf_kernel_np,
    "cosine_torch": compute_pairwise_in_batches,
    "dot_torch": compute_pairwise_in_batches,
    "euclidean_torch": compute_pairwise_in_batches,}

default_submodular_fn_params_dict = {
    "flmi": {"magnificationEta": 1},
    "flvmi": {"magnificationEta": 1},
    "gcmi": {},
    "logdetmi": {"magnificationEta": 1, "lambdaVal": 1},}
    

class SubmodularMutualInformationSelector(SubsetSelector):
    def __init__(self, 
                 encoder: BaseEncoder, 
                 device: str, 
                 similarity_fn: str,
                 submodular_fn: str, 
                 submodular_fn_params: Dict = None):
        super().__init__(encoder, device)
        self.submodular_fn = submodular_fn
        self.submodular_fn_params = submodular_fn_params
        self.similarity_fn = similarity_fn

    def select_subset(self, 
                      data, 
                      query_data,
                      subset_size: int):
        # Encode the data samples
        query_data_repr = self.encoder.encode(query_data)
        if self.submodular_fn in ["flmi", "flvmi", "logdetmi"]:
            data_repr = self.encoder.encode(data)
        

        # Initialize the similarity function
        similarity_fn = similarity_function_dict[self.similarity_fn]
        # Compute the similarity matrix
        data_query_sim_mat = similarity_fn(data_repr, query_data_repr)

        if self.submodular_fn in ["flmi", "flvmi", "logdetmi"]:
            data_sim_mat = similarity_fn(data_repr)
            if self.submodular_fn == "logdetmi":
                query_query_sim_mat = similarity_fn(query_data_repr)

        # Initialize the submodular function
        if self.submodular_fn_params is None:
            submod_fn_params = default_submodular_fn_params_dict[self.submodular_fn]
        else:
            submod_fn_params = self.submodular_fn_params

        submod_fn_params['query_sijs'] = data_query_sim_mat
        if self.submodular_fn in ["flmi", "flvmi", "logdetmi"]:
            submod_fn_params['data_sijs'] = data_sim_mat
            if self.submodular_fn == "logdetmi":
                submod_fn_params['query_query_sijs'] = query_query_sim_mat       
    
        submod_fn_params['n'] = data_repr.shape[0]
        submod_fn_params['num_queries'] = query_data_repr.shape[0]
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

                                                                     