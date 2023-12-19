# -*- coding: utf-8 -*-
from typing import Text, List
import re
from constants import CONSONANTS, SHORT, LONG, SHAMS, QAMAR, TANWEEN
from arabic_buckwalter_transliteration.transliteration import buckwalter_to_arabic, arabic_to_buckwalter
import mishkal.tashkeel
import pyarabic.araby as araby

shams = "".join(SHAMS)
#print(shams)
qamar = "".join(QAMAR)
#print(qamar)

def arabic_buckwalter(words: Text) -> List[Text]:
    """
    convert a list of arabic words into a list of buckwalter symbols
    """
    return "".join([arabic_to_buckwalter(i) for i in words])

def buckwalter_arabic(words: Text) -> List[Text]:
    """
    convert a list of arabic words into a list of buckwalter symbols
    """
    return "".join([buckwalter_to_arabic(i) for i in words])

def fix_tanween(word: Text) -> List[Text]:
    """
    Checks if the word ends in 'N' or 'K', then replaces 'N' with 'u' and 'K' with 'o'.
    words_buck_fixed = [fix_tanween(i) for i in words_buck]
    """
    if word.endswith('N'):
        return word[:-1] + 'u'
    elif word.endswith('F'):
        return word[:-1] + 'a'
    elif word.endswith('K'):
        return word[:-1] + 'i'
    else:
        return word


def check_letter_after_l(word, symbols):
    """
    # check Qamar or Shams
    Checks if the letter after 'l' in the word belongs to the specified symbols.
    # fix definite article
    shams = "".join(SHAMS)
    #print(shams)
    qamar = "".join(QAMAR)
    #print(qamar)
    check_letter_after_l("Al$~ams", shams)
    """
    # Find the index of 'l' in the word
    l_index = word.find('l')

    # If 'l' is found and it's not the last character
    if l_index != -1 and l_index < len(word) - 1:
        # Check the character after 'l'
        next_char = word[l_index + 1]
        #print(next_char)
        return next_char in symbols, next_char

    return next_char

def fix_definite_article(word):
    """
    l = [fix_definite_article(i) for i in words_buck_fixed]
    """
    if word.startswith(('Al', 'Aal')):
        l, ll = check_letter_after_l(word, shams)
        if l == True:
            x = "'a" + ll + complete_word_with_letter(word,ll)
            return x
        elif l == False:
            x = "'al" + word[2:]
            return x
        else:
            return word
    
    else:
        return word



def complete_word_with_letter(word, letter):
    """
    Completes a word starting from the given letter using regex.
    """
    # Define the pattern to match the word starting from the specified letter
    if letter in '.*+?^$()[]{}|\\':
        letter = '\\' + letter
    pattern = letter + '.*'

    # Find the match in the word
    match = re.search(pattern, word)
    if match:
        # Return the matched part of the word
        return match.group()
    else:
        return "No match found."


def get_template_from_template_bank(template_dict, template):
    """
    get_template_from_template_bank(template_bank, 'CvCVC')
    """
    for i in template_dict:
        if i['template'] == template:
            return i['syllable']




def map_arabic_to_cv_pattern_modified(word):
    """
    mapped = [map_arabic_to_cv_pattern_modified(i) for i in l]
    """
    consonants = "".join(CONSONANTS)
    short_vowels = SHORT
    long_vowel_candidates = LONG
    

    pattern = ""

    i = 0
    while i < len(word):
        char = word[i]

        if char in long_vowel_candidates:
            if i > 0 and word[i-1] in short_vowels:
                pattern = pattern[:-1]
            
            if i == len(word) - 1 or (i + 1 < len(word) and word[i + 1] not in short_vowels):
                pattern += "V"
            else:
                pattern += "C"  
            i += 1
            continue

        # Check for short vowel
        if char in short_vowels:
            pattern += "v"

        # If it's a consonant
        elif char in consonants:
            pattern += "C"

        i += 1

    return {"word": word, "template": pattern}


def split_word_arabic(word, syllable_pattern):
    syllables = []
    current_syllable = ""
    pattern_index = 0

    for char in word:
        if pattern_index >= len(syllable_pattern):
            break

        if syllable_pattern[pattern_index] in ['C', 'v']:
            current_syllable =  current_syllable + char  

        if pattern_index + 1 < len(syllable_pattern) and syllable_pattern[pattern_index + 1] == '.':
            syllables.insert(0, current_syllable)  
            current_syllable = ""
            pattern_index += 1  

        pattern_index += 1

    # Add the last syllable if it exists
    if current_syllable:
        syllables.insert(0, current_syllable)

    return syllables

def shakkal(text):
    vocalizer = mishkal.tashkeel.TashkeelClass()
    return vocalizer.tashkeel(text)

def remove_last_diacritic(word):
    if word.endswith(("َّ", "ُّ", "ِّ")):
        return word[:-2]
    elif word.endswith(("َ", "ُ", "ِ")): 
        return word[:-1]
    else:
        return word