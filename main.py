import gradio as gr
from utils.pipeline import DiffUTEPipeline
from utils.new_pipeline import MixSceneTextPipeline
from Onomatopoeia.Onomato_translation import find_similar_korean_onomatopoeia
from Onomatopoeia.input_similar import find_most_similar_word_from_OCR
import torch

from torchvision import transforms as tf
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import glob, os
import easyocr

def resize_cv2(input_image, w, h):
    output = input_image.resize((w, h))
    # output = np.array(output)
    # return output[:, :, ::-1]
    return output

def load_model():
    # pipe = DiffUTEPipeline('cuda:0')
    pipe = MixSceneTextPipeline('cuda:0')
    ckpt = torch.load('./pytorch_model.bin', map_location='cpu')
    pipe.unet.load_state_dict(ckpt)
    return pipe

def get_OCR_result(init_img):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,'
    reader = easyocr.Reader(['ko', 'en'])
    result = reader.readtext(np.array(init_img)) # PIL to cv2 (but RGB is not changed)
    
    return result

def make_mask_image(init_img, mask_text):
    result = get_OCR_result(init_img)
    
    for pts, text, p in result :
        if text == mask_text :
            mask_img = np.zeros_like(init_img)
            cv2.fillConvexPoly(mask_img, np.array(pts, dtype=np.int32), (255, 255, 255))
            break

    return Image.fromarray(mask_img), np.array(pts)

def crop_center(image, pts) :
    pts = np.array(pts) # (4, 2)
    
    cp = np.mean(pts, axis=0)
    w, h = image.size
    d = min(w, h)
    if w < h : 
        ltx, rtx = 0, d
        lty = min(max(0, cp[1]- d//2), max(0, min(cp[1] + d//2, h) - d))
        rty = lty + d
    else : 
        lty, rty = 0, d
        ltx = min(max(0, cp[0]- d//2), max(0, min(cp[0] + d//2, w) - d))
        rtx = ltx + d
    
    
    crop_pts = (int(ltx), int(lty), int(rtx), int(rty))
    return image.crop(crop_pts), crop_pts
        
def generate_image(init_img, mask_text, prompt):
    init_img = Image.fromarray(init_img)
    width, height = init_img.size

    mask_img, pts = make_mask_image(init_img, mask_text)
    
    crop_init_img, crop_pts = crop_center(init_img, pts)
    crop_mask_img, crop_pts = crop_center(mask_img, pts)
    
    w, h = crop_init_img.size
    pipe.to('diffute')
    output_image = pipe.sample(crop_init_img.resize((512, 512)), crop_mask_img.resize((512, 512)), prompt)
    output_image = output_image.resize((w, h))
    
    init_img.paste(output_image, (crop_pts[0], crop_pts[1]))
    return init_img
    # return init_img
    
def generate_image_basic(init_img, mask_img, prompt):
    init_img = Image.fromarray(init_img)
    mask_img = Image.fromarray(mask_img)
    width, height = init_img.size

    # mask_img, pts = make_mask_image(init_img, mask_text)
    
    # crop_init_img, crop_pts = crop_center(init_img, pts)
    # crop_mask_img, crop_pts = crop_center(mask_img, pts)
    
    w, h = init_img.size
    pipe.to('diffute')
    output_image = pipe.sample(init_img.resize((512, 512)), mask_img.resize((512, 512)), prompt)
    output_image = output_image.resize((w, h))
    
    # init_img.paste(output_image, (crop_pts[0], crop_pts[1]))
    return output_image
    # return init_img

def onomato_trans(input_text):
    return find_similar_korean_onomatopoeia(input_text)
    
def auto_onomato_generation(init_img, input_text):
    init_img = Image.fromarray(init_img)
    width, height = init_img.size

    mask_text = auto_Onomato_detection(init_img, input_text) # 끼기긱 -> 끼개각
    mask_img, pts = make_mask_image(init_img, mask_text) # make mask
    # prompt = find_similar_korean_onomatopoeia(mask_text)
    prompt = find_similar_korean_onomatopoeia(input_text) # Onomato 번역은 원본 텍스트로 (not 끼개각, 끼기긱)
    
    
    crop_init_img, crop_pts = crop_center(init_img, pts)
    crop_mask_img, crop_pts = crop_center(mask_img, pts)
    
    w, h = crop_init_img.size
    pipe.to('diffute')
    output_image = pipe.sample(crop_init_img.resize((512, 512)), crop_mask_img.resize((512, 512)), prompt)
    output_image = output_image.resize((w, h))
    
    init_img.paste(output_image, (crop_pts[0], crop_pts[1]))
    return mask_text, prompt, init_img

def auto_Onomato_detection(init_img, input_text):
    result = get_OCR_result(init_img)
    
    OCR_list = []
    for pts, text, p in result: 
        OCR_list.append(text)
    
    return find_most_similar_word_from_OCR(input_text, OCR_list)
    
def make_mask_coor(input_array):
    x_1, x_2 = input_array[0], input_array[2]
    y_1, y_2 = input_array[1], input_array[3]
    
    return np.array([[x_1, y_1], [x_2, y_1], [x_2, y_2], [x_1, y_2]])
    
def remove_masked_part(init_img, mask_text, prompt, font_size):
    init_img = Image.fromarray(init_img)
    init_img = init_img.resize((512, 512))
    mask_img, pts = make_mask_image(init_img, mask_text)    
    prompt_remove = "remove masked part"    
    pipe.to('inpaint')
    output_image = pipe.sample(init_img, mask_img, prompt_remove)
    
    mask_img_diffute = np.zeros_like(np.array(output_image))
    font = ImageFont.truetype("./utils/arial.ttf", font_size)
    draw = ImageDraw.Draw(output_image)
    output = draw.textbbox(pts[0], prompt, font = font)
    cv2.fillConvexPoly(mask_img_diffute, np.array(make_mask_coor(output), dtype=np.int32), (255, 255, 255))
    Image.fromarray(mask_img_diffute)
    
    pipe.to('diffute')
    result = pipe.sample(output_image, mask_img_diffute, prompt)
    return result

# Load the model
pipe = load_model()

# Define the Gradio interface
iface1 = gr.Interface(
    fn=generate_image_basic,
    inputs=["image", "image", "text"],  # Input types: two images and one text box
    outputs="image",  # Output type: one image
    title="From Init Image and Mask Image to Generated Image",
    # description="Upload an initial image and a text that wanna be masked, then enter a text prompt to generate a new image."
)

iface2 = gr.Interface(
    fn=generate_image,
    inputs=["image", "text", "text"],  # Input types: two images and one text box
    outputs="image",  # Output type: one image
    title="Image Generation with Text Prompt",
    description="Upload an initial image and a text that wanna be masked, then enter a text prompt to generate a new image."
)

iface3 = gr.Interface(
    fn=onomato_trans,
    inputs ="text",
    outputs="text",
    title= "Onomatopoeia translation into Korean."
)

iface4 = gr.Interface(
    fn = auto_onomato_generation,
    inputs = ["image", "text"],
    outputs = ["text", "text", "image"],
    title = "Auto Onomatopoeia generation."
)
iface5 = gr.Interface(
    fn = remove_masked_part,
    inputs = ["image", "text", "text", "number"],
    outputs = ["image"],
    title = "Remove + Mask making with font size"
)

tabbed_interface = gr.TabbedInterface([iface1, iface2, iface3, iface4, iface5], ["Simple Generation", "Generation", "Onomatopoeia", "Auto-Ono generation", "Just removing"])

if __name__ == "__main__":
    tabbed_interface.launch()
# tabbed_interface.launch()