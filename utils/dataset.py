from PIL import Image, ImageDraw, ImageFont

def text_image(message, dsize=256) :
    W, H = dsize, dsize
    image = Image.new('RGB', (W, H), (255, 255, 255))
    
    font = ImageFont.truetype("./utils/arial.ttf", 10)
    
    draw = ImageDraw.Draw(image)
    _, _, w, h = draw.textbbox((0, 0), message, font=font)
    
    h = int(0.8 * h * W / w)
    
    fontColor = (0, 0, 0)
    
    font = ImageFont.truetype("./utils/arial.ttf", h)
    _, _, w, h = draw.textbbox((0, 0), message, font=font)
    
    draw.text(((W-w)/2, (H-h)/2), message, font=font, fill=fontColor)
    
    return image 

def image_pad(img) :
    W, H = img.size
    D = max(W, H)
    image = Image.new('RGB', (D, D), (255, 255, 255))
    image.paste(img, ((D-W)//2, (H-W)//2))

    return image    
