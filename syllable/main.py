# -*- coding: utf-8 -*-
from utils import arabic_buckwalter, shakkal, remove_last_diacritic, buckwalter_arabic, split_word_arabic
from syllabify import Syllabifier


sentence = "منتج زراعي ثانوي حيث يتكون القش من السيقان الجافة الباقية  في من محاصيل الحبوب"

if __name__ == '__main__':
    # put diacritics
    text = shakkal(sentence).split()
    print(text)
    # remove diacritic on the last letter
    text = [remove_last_diacritic(i) for i in text]
    print(text)

    to_buckwalter = [arabic_buckwalter(word) for word in text]
    print(to_buckwalter)
    xx= []
    try:
        syl = Syllabifier()
        for word in to_buckwalter:
            xx.append(syl.syllabify(word))
    except Exception as e:
        pass
    print(xx)
    

    #x['arabic'] = buckwalter_arabic(x['word'])
    #print(x)
    #xx = split_word_arabic(x['arabic'], x['syllable'])
    #print(xx)
   