#Contains implemenation of base SubsetSelector class that uses encoder class for encoding the data samples
# and then selects the subset of samples based on the encoded representation of the samples.
# The encoder class should be a subclass of BaseEncoder class defined in encoders/base_encoder.py

class SubsetSelector:
    def __init__(self, encoder, device):
        self.encoder = encoder
        self.device = device
    
    def select_subset(self, data, subset_size):
        raise NotImplementedError
    
    