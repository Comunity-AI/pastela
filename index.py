from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import vgg16
import torch.nn.functional as F

class CustomDataset(Dataset):
    def __init__(self, image_paths, descriptions, transform=None):
        self.image_paths = image_paths
        self.descriptions = descriptions
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        description = self.descriptions[idx]

        if self.transform:
            image = self.transform(image)

        return image, description

# Configuración
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

image_paths = []
descriptions = []

with open("./datasets/prompts.txt") as f:
    for line in f:
        name, description = line.split(' - ')
        image_paths.append(f"./datasets/{name}.jpg")
        descriptions.append(description)


dataset = CustomDataset(image_paths=image_paths, descriptions=descriptions, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Cargar el modelo preentrenado
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to("cuda")

optimizer = optim.Adam(pipe.unet.parameters(), lr=5e-5)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features.eval().cuda()
        self.features = nn.Sequential(*list(vgg.children())[:16])

    def forward(self, generated, target):
        gen_features = self.features(generated)
        target_features = self.features(target)
        return F.mse_loss(gen_features, target_features)

# Usar la pérdida perceptual en el entrenamiento
loss_fn = PerceptualLoss().to("cuda")

def pil_to_tensor(image):
    transform = transforms.ToTensor()
    return transform(image).unsqueeze(0)  # Añadir dimensión del lote

# Entrenamiento
for epoch in range(1):  # Ajusta el número de épocas según sea necesario
    for images, descriptions in dataloader:
        images = images.to("cuda")

        # Tokenizar descripciones
        text_inputs = tokenizer(descriptions, return_tensors="pt", padding=True, truncation=True, max_length=77).to("cuda")

        # Convertir a lista de descripciones
        prompts = [desc for desc in descriptions]

        # Generar imágenes usando el pipeline
        outputs = pipe(prompts, num_inference_steps=50, guidance_scale=7.5).images
        
        # Convertir las imágenes generadas a tensores
        output_tensors = torch.cat([pil_to_tensor(img).to("cuda") for img in outputs])
        

        if output_tensors.size() != images.size():
            output_tensors = transforms.Resize((200, 200))(output_tensors)
        
        # Calcular pérdida
        loss = loss_fn(output_tensors, images)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.cuda.empty_cache()

# Guardar el modelo
pipe.save_pretrained('./fine_tuned_model')
