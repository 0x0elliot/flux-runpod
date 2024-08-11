import torch
from diffusers import FluxPipeline

def fetch_pretrained_model(model_class, model_name, **kwargs):
    '''
    Fetches a pretrained model from the HuggingFace model hub.
    '''
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return model_class.from_pretrained(model_name, **kwargs)
        except OSError as err:
            if attempt < max_retries - 1:
                print(f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...")
            else:
                raise

def cache_flux_model():
    '''
    Caches the FLUX.1-dev model from Black Forest Labs.
    '''
    model_name = "black-forest-labs/FLUX.1-schnell"
    print(f"Caching {model_name}...")
    
    # Cache the FLUX pipeline
    pipe = fetch_pretrained_model(FluxPipeline, model_name, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    print("FLUX pipeline cached and CPU offload enabled")
    
    return pipe

if __name__ == "__main__":

    flux_pipe = cache_flux_model()