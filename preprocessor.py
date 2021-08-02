import re
import string
import spacy

import nltk
from nltk.corpus import stopwords

class TextPreprocessor:
    def __init__(self, positive='радость', negative = 'грусть'):
        self.positive = f' {positive.strip()} '
        self.negative = f' {negative.strip()} '


    def load(self):	
        nltk.download('stopwords')
        nltk.download('punkt')
		
        russian_stopwords = stopwords.words("russian")
        russian_stopwords.remove('не')
        russian_stopwords.remove('нет')

        self.russian_stopwords = set(russian_stopwords)
        self.nlp = spacy.load('ru_core_news_sm')

    def split_hash_tag(self, text):
        match = re.search(r"#([А-Яа-я]+)", text)
        if match and match.group(1):
            replacements = ' '.join(re.findall('[А-Я][^А-Я]*', match.group(1)))
            return text.replace(match.group(0), replacements)

        return text

    def remove_parenthesis_pairs(self, text):
        text = re.sub(r'((\(|\[|\{)(.*)(\)|\]|\}))', '\g<2>', text)
        return text

    def replace_smiles(self, text):
        text = re.sub(r'((:|;|=|8)?(-|%|5|c|с)?(\)|\]|\}|3)+|😜|😄|😂|💋|♥)', self.positive, text)
        text = re.sub(r'((:|;|=|8)(-|%|5|c|с)?(d|p|\*)+)', self.positive, text)

        text = re.sub(r'((:|;|=|8)?\'?(-|%|5|c|с)?(\(|\[|\{)+)', self.negative, text)
        text = re.sub(r'((:|;|=|8)\'?(-|%|5|c|с)?(g|o)+)', self.negative, text)

        return text

    def replace_obvious_scores(self, text):
        text = re.sub(r'([7-9]|1[0-9]) из 10', self.positive, text)
        text = re.sub(r'[0-4] из 10', self.negative, text)
        return text

    def collapse_same_letters(self, text):
        text = re.sub(r'([а-яё])\1{2,}', '\g<1>', text)
        return text

    def remove_stop_words(self, text):
        words = text.split(' ')
        text = ' '.join([word for word in words if word not in self.russian_stopwords])
        return text

    def lemmatize(self, text):
        text = ' '.join([w.lemma_ for w in self.nlp(text)])
        return text

    def preprocess(self, text):
        text = self.split_hash_tag(text)
        text = text.lower()
        text = text.replace('\n', ' ')
        text = text.replace('ё', 'е')

        text = self.remove_parenthesis_pairs(text)
        text = self.replace_smiles(text)
        text = self.replace_obvious_scores(text)
        text = self.collapse_same_letters(text)

        text = self.remove_stop_words(text)

        text = re.sub(r"[^а-я ]", " ", text)
        text = re.sub(r"[а-я]{35,}", "", text)
        text = re.sub(r" {2,}", " ", text)
        text = text.strip()

        text = self.lemmatize(text)

        return text