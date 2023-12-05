from datasets import load_dataset
import qalsadi.lemmatizer
import pandas as pd
from collections import Counter
import json, string, re

lemmatizer = qalsadi.lemmatizer.Lemmatizer()
punctuation = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ''' + string.punctuation
arabic_diacritics = re.compile("""
                             ّ    | # Shadda
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)


def preprocess(text):
	translator = str.maketrans('', '', punctuation)
	text = text.translate(translator)
	text = re.sub("[0123456789]", '', text)

	# remove Tashkeel
	text = re.sub(arabic_diacritics, '', text)

	# remove elongation
	text = re.sub("[إأآا]", "ا", text)
	text = re.sub("ى", "ي", text)
	text = re.sub("ؤ", "ء", text)
	text = re.sub("ئ", "ء", text)
	text = re.sub("ة", "ه", text)
	text = re.sub("گ", "ك", text)

	text = ' '.join(word for word in text.split())

	return text


dataset = load_dataset('mozilla-foundation/common_voice_11_0','ar', split='train')
sentences = [preprocess(d['sentence']) for d in dataset]
lemmas=[]
for s in sentences:
	words = s.split()
	for w in words:
		lemmas.append(lemmatizer.lemmatize(w))

clemmas = Counter(lemmas)
json_lemmas = json.dumps(clemmas, ensure_ascii = False)

with open('arabic_lemmas.json', 'w') as f:
	f.write(json_lemmas)
