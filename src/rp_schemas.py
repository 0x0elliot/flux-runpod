INPUT_SCHEMA = {
    'prompt': {
        'type': str,
        'required': True,
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 3.5
    },
    'height': {
        'type': int,
        'required': False,
        'default': 768
    },
    'width': {
        'type': int,
        'required': False,
        'default': 1360
    },
    'num_inference_steps': {
        'type': int,
        'required': False,
        'default': 50
    },
    'seed': {
        'type': int,
        'required': False,
        'default': -1
    },
    'num_images': {
        'type': int,
        'required': False,
        'default': 1
    },
    'max_sequence_length': {
        'type': int,
        'required': False,
        'default': 256
    }
}