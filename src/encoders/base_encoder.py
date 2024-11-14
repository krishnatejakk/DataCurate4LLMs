import torch
from transformers import AutoModel, AutoTokenizer

#Base class for the sentence encoders
class BaseEncoder:
    def __init__(self, model_name, device, batch_size=512, tokenizer = False, use_fp16=False):
        if tokenizer:
            self.tokenizer, self.model = self.initialize_model_tokenizer(model_name, device, use_fp16)
        else:
            self.model = self.initialize_model(model_name, device, use_fp16)
        self.device = device
        self.batch_size = batch_size

    def initialize_model(self, model_name, device, use_fp16=False):
        # load model and tokenizer
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        if use_fp16:
            model = model.half()

        # move model to device  
        model = model.to(device)

        # Use multiple GPUs if available 
        device_count = torch.cuda.device_count()
        if device_count > 1:
            model = torch.nn.DataParallel(model)

        # set number of gpus
        self.num_gpus = device_count
        return model

    def initialize_model_tokenizer(self, model_name, device, use_fp16=False):
        # load model and tokenizer
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        if use_fp16:
            model = model.half()

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # move model to device   
        model = model.to(device)

        # Use multiple GPUs if available 
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Using {device_count} GPUs")
            model = torch.nn.DataParallel(model)

        # set number of gpus
        self.num_gpus = device_count
        return tokenizer, model
 

    def encode(self, inputs, return_tensors=False):
        pass