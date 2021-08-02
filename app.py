from flask import Flask, request
from classifier import Classifier
import datetime
import logging
logging.basicConfig(filename='classifier.log',level=logging.INFO)

app = Flask(__name__)

input_map_path = "./model/input-map.json"
input_sequence_config_path = "./model/input-sequence-length.txt"
output_map_path = "./model/output-map.json"
neural_net_config_path = "./model/model-config.json"
neural_net_weights_path = "./model/model-weights.h5"
embeddings_path = './ru/ru.wiki.bpe.vs100000.d100.w2v.txt'

classifier = Classifier(neural_net_config_path, neural_net_weights_path,
                                   input_map_path, input_sequence_config_path,
                                   output_map_path, embeddings_path)

print('loading started...')
classifier.load()
print('ready!')

@app.route('/')
def check_server():
    return {'server_status': 'running'}

@app.route('/classify', methods=['Post'])
def classify_text():
     try:
         text = request.form['text']
         app.logger.info(f'Classification requested: {text}')
         start_time = datetime.datetime.now()

         classification_result = classifier.classify(text)
         result = int(classification_result[0][0])
         if result < 0:
             string_result = 'negative'
         elif result > 0:
             string_result = 'positive'
         else:
             string_result = 'neutral'

         end_time = datetime.datetime.now()
         execution_time_ms = (end_time - start_time).total_seconds() * 1000

         app.logger.info(f'Classification result: {classification_result}, execution time, ms: {execution_time_ms}')
         return {'result': string_result}

     except Exception as error:
         print(error)
         app.logger.error(error)
