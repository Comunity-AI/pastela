import torch
from matplotlib import pyplot as plt
from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, PNDMScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"

# Reconstruir el pipeline
pipe = pipe = StableDiffusionPipeline.from_pretrained('./models/fine_tuned_model').to(device)

prompt = "fondo estilo pastel de " + "unas montañas verdes"
num_inference_steps = 40
guidance_scale = 7.5

# Generar la imagen
with torch.no_grad():
    generated_images = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images

# Mostrar la imagen
from PIL import Image
generated_images[0].save("./imagen.png")
# Mostrar la primera imagen generada
generated_images[0].show()

