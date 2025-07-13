from diffusers import FlowMatchEulerDiscreteScheduler
import torch
from step import StableDiffusion3Pipeline  
import pandas as pd

torch.cuda.empty_cache()
def run_inference(device, model_id, inference_steps):
    torch.cuda.empty_cache()
    prompt = "A pretty girl with anime style"
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16, 
    )
    pipe = pipe.to(device)
    pipe.scheduler = scheduler
    # Generate origrinal image
    gen1 = torch.Generator(device).manual_seed(0)
    images = pipe(prompt, num_inference_steps = inference_steps, l = inference_steps/4, r = inference_steps, device = device, generator = gen1).images  
    images[0].save(f"org_{inference_steps}steps.png")
    # Caluate w_g 
    # To show the convergence if w_g, we choose different seed
    gen2 = torch.Generator(device).manual_seed(1)
    images = pipe(prompt, num_inference_steps = inference_steps, cal_wg = True, skip_x = False, l = inference_steps/4, r = inference_steps, device = device, generator = gen2).images 
    # LTC-Accel process. l and r control the accelerate interval 
    gen3 = torch.Generator(device).manual_seed(0)
    images = pipe(prompt, num_inference_steps = inference_steps, cal_wg = False, skip_x = True, l = inference_steps/4, r = inference_steps, device = device, generator = gen3).images 
    images[0].save(f"LTC-Accel_{inference_steps}steps.png")

if __name__ == "__main__":
    #sd35
    model_id = "stabilityai/stable-diffusion-3.5-large"
    
    run_inference(device="cuda", model_id=model_id, inference_steps=40)


