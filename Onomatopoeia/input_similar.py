import pandas as pd
from Levenshtein import distance as lev_distance
from Onomatopoeia.Onomato_translation import decompose_string


def find_most_similar_word_from_OCR(input_word, words_list):
    min_distance = float('inf')
    most_similar_word = None

    decomposed_input = decompose_string(input_word)

    for word in words_list:
        decomposed_word = decompose_string(word)
        distance = lev_distance(decomposed_input, decomposed_word)

        if distance < min_distance:
            min_distance = distance
            most_similar_word = word

    return most_similar_word
