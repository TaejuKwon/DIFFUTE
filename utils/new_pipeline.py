    
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, DDIMScheduler
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import CLIPTextModel, CLIPTokenizer

from torch.utils.data import DataLoader

import torch.nn.functional as F

from torchvision import transforms
from PIL import Image

import torch
from tqdm.auto import tqdm

from utils.dataset import text_image

class MixSceneTextPipeline :
    def __init__(self, device='cuda:0', diffute_ckpt_pth='/mnt/c/Users/USER/LABis/DiffUTE/pytorch_model.bin') :
        pretrained_model = 'runwayml/stable-diffusion-inpainting'
                
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model, subfolder='unet', revision=None)
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model, subfolder="vae", revision=None)
        self.noise_scheduler = DDIMScheduler.from_pretrained(
            pretrained_model, subfolder="scheduler")
        
        self.mode = 'inpaint'
        
        # 1. Inpainting Checkpoint 
        self.inpainting_ckpt = self.unet.state_dict()
    
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model, subfolder="text_encoder")
        
        # 2. DiffUTE Checkpoint
        self.diffute_ckpt = torch.load(diffute_ckpt_pth, map_location='cpu')
        
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
        self.glyph_encoder = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
    
    
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.glyph_encoder.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        self.unet.to(device, dtype=torch.float16)
        self.vae.to(device, dtype=torch.float16)
        self.glyph_encoder.to(device, dtype=torch.float16)
        self.text_encoder.to(device, dtype=torch.float16)
        
        self.device = device
        self.preprocess = transforms.ToTensor()
    
    def to(self, mode='diffute') :
        self.mode = mode
        if mode == 'diffute' :
            self.unet.load_state_dict(self.diffute_ckpt)
        elif mode == 'inpaint' :
            self.unet.load_state_dict(self.inpainting_ckpt)
        
        self.unet.requires_grad_(False)
        self.unet.to(self.device, dtype=torch.float16)
    
    def sample(self, init_image, mask_image, prompt, timestamps=50) :
        #### image context
        init_tensor = self.preprocess(init_image).unsqueeze(0).to(self.device, dtype=torch.float16) * 2 - 1.0
        mask_tensor = self.preprocess(mask_image).unsqueeze(0).to(self.device, dtype=torch.float16)
        covr_tensor = init_tensor * (mask_tensor < 0.5)
        mask_tensor = mask_tensor[:, 0, :, :].unsqueeze(1)
        
        covr_latents = self.vae.config.scaling_factor * self.vae.encode(covr_tensor).latent_dist.sample()
        h, w = covr_latents.shape[2:]

        mask_latents = F.interpolate(mask_tensor, size=(h, w))

        #### glyph context
        if self.mode == 'diffute' :
            glyph_image = text_image(prompt, 256).convert('RGB')
            glyph_patch = self.processor(glyph_image, return_tensors="pt").pixel_values.to(self.device, dtype=torch.float16)
            context = self.glyph_encoder.encoder(glyph_patch).last_hidden_state 
        elif self.mode == 'inpaint' :
            text_input = self.tokenizer(
                [prompt],
                max_length=self.tokenizer.model_max_length, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            ).input_ids.to(self.device)

            with torch.no_grad() :
                context = self.text_encoder(text_input)[0].to(dtype=torch.float16)
            
        self.noise_scheduler.set_timesteps(timestamps)
        
        sample_size = self.unet.config.sample_size

        batch_size = init_tensor.shape[0]
        latents = torch.randn(
            (batch_size, 4, sample_size, sample_size),
        )

        latents = latents.to(self.device, dtype=torch.float16)
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
                    encoder_hidden_states=context,
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