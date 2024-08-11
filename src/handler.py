import runpod
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
model_name = "black-forest-labs/FLUX.1-schnell"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7
        )
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

def handler(job):
    job_input = job['input']
    prompt = job_input.get('prompt', '')
    
    if len(prompt) == 0:
        return {"error": "Prompt is required"}
    
    max_length = job_input.get('max_length', 100)
    
    generated_text = generate_text(prompt, max_length)
    return {"generated_text": generated_text}

runpod.serverless.start({"handler": handler})