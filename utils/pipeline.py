from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, DDIMScheduler
from transformers import CLIPTextModel, AutoTokenizer
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from torch.utils.data import DataLoader

import torch.nn.functional as F

from torchvision import transforms
from PIL import Image

import torch
from tqdm.auto import tqdm

from utils.dataset import text_image

class DiffUTEPipeline :
    def __init__(self, device='cuda:0') :
        pretrained_model = 'runwayml/stable-diffusion-inpainting'
                
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model, subfolder='unet', revision=None)
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model, subfolder="vae", revision=None)
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model, subfolder="scheduler")
        
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
        self.glyph_encoder = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
    
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.glyph_encoder.requires_grad_(False)

        self.unet.to(device)
        self.vae.to(device)
        self.glyph_encoder.to(device)
        
        self.device = device
        self.preprocess = transforms.ToTensor()
    
    def sample(self, init_image, mask_image, prompt) :
        #### image context
        init_tensor = self.preprocess(init_image).unsqueeze(0).to(self.device) * 2 - 1.0
        mask_tensor = self.preprocess(mask_image).unsqueeze(0).to(self.device)
        covr_tensor = init_tensor * (mask_tensor < 0.5)
        mask_tensor = mask_tensor[:, 0, :, :].unsqueeze(1)
        
        covr_latents = self.vae.config.scaling_factor * self.vae.encode(covr_tensor).latent_dist.sample()
        h, w = covr_latents.shape[2:]

        mask_latents = F.interpolate(mask_tensor, size=(h, w))

        #### glyph context
        glyph_image = text_image(prompt, 256).convert('RGB')
        glyph_patch = self.processor(glyph_image, return_tensors="pt").pixel_values.to(self.device)
        glyph_context = self.glyph_encoder.encoder(glyph_patch).last_hidden_state 
                
        self.noise_scheduler.set_timesteps(50)
        
        sample_size = self.unet.config.sample_size

        batch_size = init_tensor.shape[0]
        latents = torch.randn(
            (batch_size, 4, sample_size, sample_size),
        )

        latents = latents.to(self.device)
        latents = latents * self.noise_scheduler.init_noise_sigma

        for t in tqdm(self.noise_scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = latents

            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, timestep=t)
            latent_model_input = torch.cat([latent_model_input, mask_latents, covr_latents], dim=1)

            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input, 
                    t, 
                    encoder_hidden_states=glyph_context,
                ).sample

        
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        ### 4. Sample to Image
        scaled_latents = latents / self.vae.config.scaling_factor
        with torch.no_grad():
            image = self.vae.decode(scaled_latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        
        return pil_images[0]
    