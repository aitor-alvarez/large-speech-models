# -*- coding: utf-8 -*-
from typing import List, Text
from datasets import load_dataset
from bank import template_bank
from utils import map_arabic_to_cv_pattern_modified, get_template_from_template_bank, fix_definite_article




class Syllabifier:
    def __init__(self):
        self.prefixes = ("Aal", "Al") # tuple of prefixes
        #self.suffixes = ("k") # tuple of suffixes
        self.fun = ["min"]
        

    def syllabify(self, word):
        """call the previous three functions"""
        if word.startswith(self.prefixes):
            return self.syllabify_prefix(word)
        elif word in self.fun:
            return self.syllabify_fun(word)
        else:
            return self.syllabify_suffix(word)

    def syllabify_suffix(self, word):
        """syllabify all suffixes"""
        """possessives added to nouns. final with no diacritics. """
        if word.endswith("i"):
            # 1) look original word in bank, and return the syllabus
            original = word[:-1]
            # 2) map syllables and vowels
            original = map_arabic_to_cv_pattern_modified(original)
            #print(original)
            # get syllable structure from the bank
            res = get_template_from_template_bank(template_bank, original['template'])
            res = res + 'V'
            a = res.split(".")
            if a[-1] == "CvCV" or "CVCv" or "CvC": # include both short and long
                x = a[-1][0:2] + "." + a[-1][2:]
                a.append(x)
            a.pop(-2)
            return {"word":word, "syllable":".".join(a)}
        
        elif word.endswith("k") or word.endswith("h"):
            original = word[:-2]
            original = map_arabic_to_cv_pattern_modified(original)
            print(original)
            res = get_template_from_template_bank(template_bank, original['template'])
            print(res)
            res = res + 'vC'
            a = res.split(".")
            if a[-1] == "CVCvC" or "CvCvC": # include both short and long
                x = a[-1][0:2] + "." + a[-1][2:]
                a.append(x)
            a.pop(-2)
            return {"word":word, "syllable":".".join(a)}
        
        elif word.endswith("ha") or word.endswith("haA")  or word.endswith("na") or word.endswith("naA"):
            original = word[:-4]
            original = map_arabic_to_cv_pattern_modified(original)
            print(original)
            res = get_template_from_template_bank(template_bank, original['template'])
            res = res + 'vCV'
            a = res.split(".")
            if a[-1] == "CVCvCV" or "CVCVCV" or "CvCvCV" or "CVCVCV":
                x = a[-1][0:2] + "." + a[-1][2:4] + "." + a[-1][4:]
                a.append(x)
            a.pop(-2)
            return {"word":word, "syllable":".".join(a)}
        
        elif word.endswith("kum") or word.endswith("hum"):
            original = word[:-4]
            original = map_arabic_to_cv_pattern_modified(original)
            res = get_template_from_template_bank(template_bank, original['template'])
            res = res + 'vCvC'
            a = res.split(".")
            if a[-1] == "CVCvCvC" or "CvCvCvC" or "CvCVCvC" or "CvCvCVC":
                x = a[-1][0:2] + "." + a[-1][2:4] + "." + a[-1][4:]
                a.append(x)
                a.pop(-2)
            return {"word":word, "syllable":".".join(a)}
        
        elif word.endswith("t"): # past tense
            original = word[:-1]
            original = map_arabic_to_cv_pattern_modified(original)
            res = get_template_from_template_bank(template_bank, original['template'])
            res = res + 'C'
            return {"word":word, "syllable":res}
        
        elif word.endswith("wA"): # past tense
            original = word[:-3]
            original = map_arabic_to_cv_pattern_modified(original)
            res = get_template_from_template_bank(template_bank, original['template'])
            res = res + 'V'
            a = res.split(".")
            if a[-1] == "CvCV":
                x = a[-1][0:2] + "." + a[-1][2:] 
                a.append(x)
                a.pop(-2)
            return {"word":word, "syllable":".".join(a)}
        
        elif word.endswith("tum"): # past tense
            original = word[:-3]
            original = map_arabic_to_cv_pattern_modified(original)
            res = get_template_from_template_bank(template_bank, original['template'])
            res = res + 'CvC'
            a = res.split(".")
            if a[-1] == "CvCCvC":
                x = a[-1][0:3] + "." + a[-1][3:] 
                a.append(x)
                a.pop(-2)
            return {"word":word, "syllable":".".join(a)}
        
        else:
            original = map_arabic_to_cv_pattern_modified(word)
            res = get_template_from_template_bank(template_bank, original['template'])
            return {"word":word, "syllable":res}
            
            

            

    def syllabify_prefix(self, word):
        """syllabify prefixes"""
        if word.startswith((">", "t", "y", "n")) and word.startswith(("iy", "wA")):
            original = map_arabic_to_cv_pattern_modified(word)
            #FIXME: يدرسوا - تدرسي do not work
            return original
        elif word.startswith((">", "t", "y", "n")):
            original = map_arabic_to_cv_pattern_modified(word)
            res = get_template_from_template_bank(template_bank, original['template'])
            return {"word":word, "syllable":res}
        
        
        elif word.startswith(("Al", "Aal")): 
            fix_al = fix_definite_article(word) 
            if fix_al.startswith(("Al", "Aal")): # QAMAR AL
                mapping = map_arabic_to_cv_pattern_modified(fix_al)
                # look template with Al in template bank
                temp = mapping['template'][3:]
                res = get_template_from_template_bank(template_bank, temp)
                a = "CvC." + res
                return {"word":word, "syllable":a}
            elif fix_al.startswith(("'as", "Asl")): # SHAMS AL
                # remove shadda
                mapping = map_arabic_to_cv_pattern_modified(fix_al)
                temp = mapping['template'][4:]
                res = get_template_from_template_bank(template_bank, temp) 
                a = "CvC.C" + res
                return {"word":mapping['word'], "syllable":a}
            else:
                return word

        else:
            original = map_arabic_to_cv_pattern_modified(word)
            res = get_template_from_template_bank(template_bank, original['template'])
            return {"word":word, "syllable":res}

    def syllabify_fun(self, word):
        """syllabify functional words"""
        if word in self.fun:
            original = map_arabic_to_cv_pattern_modified(word)
            res = get_template_from_template_bank(template_bank, original['template'])
            return {'word': word, 'syllable': res}
            #return res


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

    if current_syllable:
        syllables.insert(0, current_syllable)

    return syllables











