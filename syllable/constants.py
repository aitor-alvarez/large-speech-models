

CONSONANTS = [
    u'b', 
    u't', 
    u'^', 
    u'j', 
    u'H', 
    u'x', 
    u'd', 
    u'*', 
    u'r', 
    u'z', 
    u's', 
    u'$', 
    u'S', 
    u'D', 
    u'T', 
    u'Z'
    u'E', 
    u'g', 
    u'f', 
    u'q', 
    u'k', 
    u'l', 
    u'm', 
    u'n', 
    u'h',
    # Hamazat
    u'\'', # ء
    u'>', # أ
    u'<', # إ
    u'&', # ؤ
    u'}', # ئ
    # Taa Marbouta
    u'p'
]
x = "".join(CONSONANTS)
print(x)
SHORT = [u'a', u'u', u'i', u'X']
TANWEEN = [u'F', u'K', u'N']
LONG = [u'A', u'y', u'w']
SHADDA = u'~'
MADDA = u'|'
SUKUN = u'o'
ALIFMAQSOURA = u'Y'
TATWEEL = u'ı'

OTHER = [u'C', u'V', u'G', u'L', u'P', u'O', u'e']# 'چ ڤ گ ڵ پ ۆ ێ'

SHAMS = [u't', u'^', u'd', u'*', u'r', u'z', u's', u'$', u'S', u'D', u'T', u'Z', u'l', u'n']


QAMAR = [u'\'', u'>', u'<', u'&', u'}', u'b', u'j', u'H', u'x', u'E', u'g', u'f', u'q', u'k', u'm', u'h', u'y', u'w']


a = {
    u'b': u'\u0628', u'*': u'\u0630', u'T': u'\u0637', u'm': u'\u0645',
    u't': u'\u062a', u'r': u'\u0631', u'Z': u'\u0638', u'n': u'\u0646',
    u'^': u'\u062b', u'z': u'\u0632', u'E': u'\u0639', u'h': u'\u0647',
    u'j': u'\u062c', u's': u'\u0633', u'g': u'\u063a', u'H': u'\u062d',
    u'q': u'\u0642', u'f': u'\u0641', u'x': u'\u062e', u'S': u'\u0635',
    u'$': u'\u0634', u'd': u'\u062f', u'D': u'\u0636', u'k': u'\u0643',
    u'>': u'\u0623', u'\'': u'\u0621', u'}': u'\u0626', u'&': u'\u0624',
    u'<': u'\u0625', u'|': u'\u0622', u'A': u'\u0627', u'Y': u'\u0649',
    u'p': u'\u0629', u'y': u'\u064a', u'l': u'\u0644', u'w': u'\u0648',
    u'F': u'\u064b', u'N': u'\u064c', u'K': u'\u064d', u'a': u'\u064e',
    u'u': u'\u064f', u'i': u'\u0650', u'~': u'\u0651', u'o': u'\u0652',
    u'C': u'\u0686', u'G': u'\u06AF', u'P': u'\u067E', u'ı': u'\u0640',
    u'V': u'\u06A4', u'L': u'\u06B5', u'O': u'\u06C6', u'e': u'\u06CE'
}
