import string, re

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
                             ـ     # Tatwil/Kashid
                         """, re.VERBOSE)


def process_text(text):
	translator = str.maketrans('', '', punctuation)
	text = text.translate(translator)
	text = re.sub("[0123456789]", '', text)

	#Do not use for Whisper. If using Whisper, comment from here
	# remove Tashkeel
	text = re.sub(arabic_diacritics, '', text)
	# Until here

	# remove elongation
	text = re.sub("[إأآا]", "ا", text)
	text = re.sub("ى", "ي", text)
	text = re.sub("ؤ", "ء", text)
	text = re.sub("ئ", "ء", text)
	text = re.sub("ة", "ه", text)
	text = re.sub("گ", "ك", text)


	text = ' '.join(word for word in text.split())

	return text


