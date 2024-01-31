from torchvision import transforms as tf
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

import pandas as pd
from Levenshtein import distance as lev_distance

import glob, os
import easyocr
from Onomatopoeia.Onomato_translation import decompose_string

def get_OCR_result_for_d(init_img):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,'
    reader = easyocr.Reader(['ko', 'en'])
    result = reader.readtext(np.array(init_img)) # PIL to cv2 (but RGB is not changed)
    
    return result

def find_most_similar_word_from_OCR(input_word):
    df = pd.read_csv('/mnt/c/Users/USER/LABis/DiffUTE/Onomatopoeia/korean_English_Onomatopoeia.csv')
    english_onomatopoeia_list = df['Korean Onomatopoeia'].tolist()
    min_distance = float('inf')
    max_distance = 0
    decomposed_input = decompose_string(input_word)

    for word in english_onomatopoeia_list:
        decomposed_word = decompose_string(word)
        distance = lev_distance(decomposed_input, decomposed_word)
        if distance < min_distance:
            min_distance = distance
            
    for word in english_onomatopoeia_list:
        decomposed_word = decompose_string(word)
        distance = lev_distance(decomposed_input, decomposed_word)
        if distance > max_distance:
            max_distance = distance

    return min_distance/max_distance

def get_background_color(sample_image, input_text):
    image = sample_image # need to be changed
    # Crop a small region around the text
    margin = 5  # pixels
    result = get_OCR_result_for_d(sample_image)
    for pts, t, p in result:
        if input_text == t:
            coordinates = pts
            break
    x_min, y_min = min(coordinates, key=lambda x: x[0])[0] - margin, min(coordinates, key=lambda x: x[1])[1] - margin
    x_max, y_max = max(coordinates, key=lambda x: x[0])[0] + margin, max(coordinates, key=lambda x: x[1])[1] + margin
    cropped_region = image[y_min:y_max, x_min:x_max]

    # Calculate the most common color
    avg_color = np.mean(cropped_region, axis=(0, 1))
    r_prob = avg_color[0] / 255
    g_prob = avg_color[1] / 255
    b_prob = avg_color[2] / 255
    return r_prob, g_prob, b_prob

def get_OCR_prob(input_text, sample_image):
    result = get_OCR_result_for_d(sample_image)
    for pts, text, p in result:
        if input_text == text:
            output = p
    return output

def get_probs_for_text(text, sample_image):
    result = get_OCR_result_for_d(sample_image)
    r_prob, g_prob, b_prob = get_background_color(sample_image, text)
    ocr_prob = get_OCR_prob(text, sample_image)
    onomatopoeia_prob = find_most_similar_word_from_OCR(text)
    return np.array([r_prob, g_prob, b_prob, ocr_prob, onomatopoeia_prob])

def concated_prob_from_image(image):
    result = get_OCR_result_for_d(image)
    
    list_t = []
    for i in range(len(result)):
        list_t.append(result[i][1])
    output = []
    for t in list_t:
        output.append(np.mean(get_probs_for_text(t, image)))
        
    return list_t, output