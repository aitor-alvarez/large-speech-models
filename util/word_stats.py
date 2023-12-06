from datasets import load_dataset
import qalsadi.lemmatizer
from collections import Counter
import json, re
from util.arabic_preprocess import process_text

lemmatizer = qalsadi.lemmatizer.Lemmatizer()



dataset = load_dataset('mozilla-foundation/common_voice_13_0','ar', split='train')
sentences = [process_text(d['sentence']) for d in dataset]
lemmas=[]
for s in sentences:
	words = s.split()
	for w in words:
		lemmas.append(lemmatizer.lemmatize(w))

clemmas = Counter(lemmas)
json_lemmas = json.dumps(clemmas, ensure_ascii = False)

with open('arabic_lemmas.json', 'w') as f:
	f.write(json_lemmas)
