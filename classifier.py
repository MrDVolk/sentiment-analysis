import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bpemb import BPEmb
from pathlib import Path
import operator

from preprocessor import TextPreprocessor


class Classifier:
    def __init__(self, neural_net_config_path, neural_net_weights_path,
                 input_map_path, input_sequence_config_path, output_map_path,
                 embeddings_path):
        self.neural_net_config_path = neural_net_config_path
        self.neural_net_weights_path = neural_net_weights_path
        self.input_map_path = input_map_path
        self.input_sequence_config_path = input_sequence_config_path
        self.output_map_path = output_map_path
        self.embeddings_path = embeddings_path

    def load(self):
        with open(self.neural_net_config_path, 'r', encoding='utf-8') as file:
            json_model = file.read()

        model = tf.keras.models.model_from_json(json_model)
        model.load_weights(self.neural_net_weights_path)

        with open(self.output_map_path, 'r', encoding='utf-8') as file:
            output_map_json = file.read()

        with open(self.input_sequence_config_path, 'r', encoding='utf-8') as file:
            self.max_sequence_length = int(file.read())

        reversed_output_map = json.loads(output_map_json)
        output_map = {v: k for k, v in reversed_output_map.items()}

        self.model = model
        self.output_map = output_map

        self.text_preprocessor = TextPreprocessor()
        self.text_preprocessor.load()

        embeddings_keys = []
        with open(self.embeddings_path, 'r', encoding='utf-8') as file:
            for line in file:
                values = line.split()
                word = values[0].lower()
                embeddings_keys.append(word)

        self.bpemb = BPEmb(lang='ru', cache_dir=Path('./'), dim=100, vs=100000)

        tokenizer = Tokenizer(len(embeddings_keys))
        tokenizer.fit_on_texts(embeddings_keys)

        del embeddings_keys[:]
        del embeddings_keys

        self.tokenizer = tokenizer

    def get_vector(self, text):
        text = self.text_preprocessor.preprocess(text)
        seq = self.tokenizer.texts_to_sequences([self.bpemb.encode(text)])
        padded = pad_sequences([seq], self.max_sequence_length)[0]

        result = np.array(padded.T)
        return result

    def classify(self, input_text):
        vector = self.model.predict([self.get_vector(input_text)]).T
        vector = np.concatenate(vector, axis=0 )
        zipped = np.array(list(zip(self.output_map.keys(), vector.T)))
        answer = {x[0]:x[1] for x in zipped}
        answer = sorted(answer.items(), key=lambda x: float(operator.itemgetter(1)(x)), reverse=True)[:10]

        return answer