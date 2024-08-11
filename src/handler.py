import runpod
import torch
import os
import shutil
import base64
from diffusers import FluxPipeline
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from rp_schemas import INPUT_SCHEMA

# Load Flux pipeline
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

def _setup_generator(seed):
    generator = torch.Generator(device="cuda")
    if seed != -1:
        generator.manual_seed(seed)
    return generator

def save_and_encode_images(images, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_base64_list = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path)
        
        # Read the image file and encode it to base64
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        image_base64_list.append(encoded_string)
    
    # Clean up the temporary directory
    shutil.rmtree(f"/{job_id}")
    
    return image_base64_list

def generate_image(job):
    '''
    Generate an image from text using Flux
    '''
    job_input = job["input"]

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']

    generator = _setup_generator(validated_input['seed'])

    # Generate image
    output = pipe(
        prompt=validated_input['prompt'],
        guidance_scale=validated_input['guidance_scale'],
        height=validated_input['height'],
        width=validated_input['width'],
        num_inference_steps=validated_input['num_inference_steps'],
        generator=generator,
        num_images_per_prompt=validated_input['num_images'],
    ).images
        
    image_urls = save_and_encode_images(output, job['id'])

    return {"image_url": image_urls[0]} if len(image_urls) == 1 else {"images": image_urls}


runpod.serverless.start({"handler": generate_image})