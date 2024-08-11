import os
import torch
from diffusers import FluxPipeline
import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

INPUT_SCHEMA = {
    "prompt": {
        "type": str,
        "required": True
    },
    "num_inference_steps": {
        "type": int,
        "required": False,
        "default": 4,
        "min": 1,
        "max": 50
    },
    "seed": {
        "type": int,
        "required": False,
        "default": 42
    }
}

def load_model():
    model_id = "black-forest-labs/FLUX.1-schnell"
    MODEL = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    MODEL.enable_model_cpu_offload()
    return MODEL

MODEL = load_model()

def run(job):
    job_input = job['input']
    
    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)
    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']
    
    # Set seed
    if validated_input['seed'] is None:
        validated_input['seed'] = int.from_bytes(os.urandom(2), "big")
    
    # Generate image
    generator = torch.Generator("cpu").manual_seed(validated_input['seed'])
    image = MODEL(
        validated_input["prompt"],
        output_type="pil",
        num_inference_steps=validated_input["num_inference_steps"],
        generator=generator
    ).images[0]
    
    # Save and upload image
    temp_file = f"/tmp/{job['id']}_output.png"
    image.save(temp_file)
    image_url = rp_upload.upload_image(job['id'], temp_file)
    
    os.remove(temp_file)
    
    return {
        "image_url": image_url,
        "seed": validated_input['seed']
    }
    
runpod.serverless.start({"handler": run, "startup_timeout": 300})