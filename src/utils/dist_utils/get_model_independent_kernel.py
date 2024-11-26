import torch
from src.encoders.bge_unified_encoder import UnifiedBGEEncoder as BGEEncoder
from src.encoders.sentence_encoder import SentenceEncoder
from src.encoders.sfr_mistral_encoder import SFRMistralEncoder
from src.utils.compute_pairwise_similarity import compute_pairwise_dense, compute_pairwise_sparse
from transformers import AutoTokenizer

def generate_instruction(icl=False, response_type="detailed and informative", 
                         considerations="accuracy, clarity, and context"):
    base_instruction = f"Provide a {response_type} response to the following user query. Consider {considerations} in your answer."
    
    if icl:
        return f"""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. You will be presented with an example of user query and the corresponding assistant response, followed by a new user query. The given example serve as a guide for the structure, style, and depth of your response. Pay close attention to:

1. The format and organization of the response
2. The level of detail provided
3. Any specific patterns or techniques used in addressing the query

After the example, you will receive a new user query. Apply the insights gained from the example to formulate your response, while also adhering to the following guideline:

{base_instruction}

Remember, while the example is meant to guide you, each user query is unique. Tailor your response to the specific needs of the new user query while maintaining the general approach demonstrated in the example.

Example:
"""
    else:
        return "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. " + base_instruction

class ModelIndependentUtility:
    def __init__(self, model_name, tokenizer_name, 
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device  # Store the device for later use
        
        # Initialize the encoder based on the model name
        if model_name.startswith("BAAI"):
            self.encoder = BGEEncoder(model_name=model_name,
                                      device=device)
        elif model_name.startswith("Salesforce"):
            self.encoder = SFRMistralEncoder(model_name=model_name,
                                             device=device)
        else:
            self.encoder = SentenceEncoder(model_name=model_name,
                                           device=device)

        # Initialize the tokenizer    
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, 
                                                       use_fast=True)

        # Set default ChatML template if tokenizer has no set chat template
        if not hasattr(self.tokenizer, 'chat_template') or self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""

    def compute_utility(self, prompts, responses):
        """
        Compute the utility of responses based on the prompts.

        Args:
            prompts (List[str]): List of prompts.
            responses (List[str]): List of responses.

        Returns:
            torch.Tensor: Utility scores for the responses.
        """

        formatted_conversations = []
        for prompt, response in zip(prompts, responses):
            instruction = generate_instruction(icl=False)
            
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            
            # Use the render_chat_template method instead of tokenizer.apply_chat_template
            formatted_text = self.tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=False,
                            )
            
            formatted_conversations.append(formatted_text)

        self.encodings = self.encoder.encode(formatted_conversations, return_tensors=True)

        # Compute the pairwise similarities between prompts and responses
        utility_scores = self.compute_pairwise_similarities(self.encodings, device=self.device)

        return utility_scores

    def compute_pairwise_similarities(self, tensor1, tensor2=None, 
                                    sparse=False, num_neighbors=100, metric='cosine', 
                                    batch_size=10000, scaling='additive', kw=0.1, 
                                    n_list=100, use_inverse_index=False, device=None):
        """
        Compute pairwise similarities between rows of two arrays, either using dense or sparse representation.

        Parameters:
            tensor1 (torch.Tensor): First matrix.
            tensor2 (torch.Tensor, optional): Second matrix. If None, uses tensor1.
            sparse (bool): If True, use sparse computation. Otherwise, use dense computation.
            num_neighbors (int): Number of neighbors (used in sparse computation).
            metric (str): The metric to use ('cosine', 'dot', 'euclidean').
            batch_size (int): Size of each batch for dense computation.
            scaling (str, optional): Type of scaling to apply in dense computation.
            kw (float, optional): Kernel width for rbf metric.
            n_list (int, optional): Number of list for Faiss Inverse Index Building.
            device (str, optional): Device to perform computation ('cuda' or 'cpu').

        Returns:
            torch.Tensor: Matrix representing pairwise similarities.
        """
        if device is None:
            device = self.device
        if sparse:
            return compute_pairwise_sparse(tensor1, tensor2, num_neighbors=num_neighbors, 
                                        batch_size=batch_size, metric=metric, scaling=scaling, 
                                        kw=kw, n_list=n_list, use_inverse_index=use_inverse_index, device=device)
        else:
            return compute_pairwise_dense(tensor1, tensor2, batch_size=batch_size, 
                                        metric=metric, device=device, scaling=scaling, 
                                        kw=kw)

if __name__ == "__main__":
    # Test the ModelIndependentUtility class
    # model_name = "BAAI/bge-large-en-v1.5"
    model_name = 'Salesforce/SFR-Embedding-Mistral'
    tokenizer_name = "mistralai/Mistral-7B-v0.1"
    
    # Initialize the ModelIndependentUtility
    utility_calculator = ModelIndependentUtility(model_name, tokenizer_name)
    
    # Sample prompts and responses
    prompts = [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "How do I bake a chocolate cake?"
    ]
    responses = [
        "The capital of France is Paris, a global center for art, fashion, gastronomy, and culture.",
        "The theory of relativity, proposed by Albert Einstein, describes how the laws of physics are the same for all non-accelerating observers, and shows that the speed of light within a vacuum is the same no matter the speed at which an observer travels.",
        "To bake a chocolate cake, you'll need ingredients like flour, sugar, cocoa powder, eggs, and milk. Mix dry and wet ingredients separately, then combine. Pour into a greased pan and bake at 350°F for about 30 minutes."
    ]
    
    # Compute utility scores
    utility_scores = utility_calculator.compute_utility(prompts, responses)
    
    # Print results
    print("Utility Scores:", utility_scores)
    