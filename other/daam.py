from daam import trace, set_seed
from diffusers import StableDiffusionPipeline
from matplotlib import pyplot as plt
import torch


model_id = 'runwayml/stable-diffusion-inpainting'
device = 'cuda'

pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

prompt = 'a face'

with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
    with trace(pipe) as tc:
        out = pipe(prompt, num_inference_steps=30)
        heat_map = tc.compute_global_heat_map()
        heat_map = heat_map.compute_word_heat_map('a face')
        heat_map.plot_overlay(out.images[0])
        plt.show()
