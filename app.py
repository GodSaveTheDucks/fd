from flask import Flask, request, jsonify
from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config
from rasa_nlu.model import Trainer, Metadata, Interpreter
#from simpletransformers.question_answering import QuestionAnsweringModel

import rasa_nlu
import spacy
import numpy as np

def trainModel(data, model_name):
    training_data = load_data(data)
    trainer = Trainer(config.load("config.yml"))
    trainer.train(training_data)
    model_directory = trainer.persist('./models/nlu/', fixed_model_name=model_name)
    interpreter = Interpreter.load(model_directory)
    return interpreter, model_directory

app = Flask(__name__)
aug_interpreter, aug_mod_dir = trainModel('models/intent.json','augmented')
nlp = spacy.load('en')
  
# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.
@app.route('/classifyContext',methods=['POST'])
def classify_context(question, intent):
    if intent == 'Delivery':
        model_directory = 'models/base_context/'
        interpreter = Interpreter.load(model_directory) 
        context = interpreter.parse(question)['intent']['name']
        return context


@app.route('/classifyIntent',methods = ['POST'])
# ‘/’ URL is bound with hello_world() function.
def classifyIntent():
    '''
    params :  data <obj> - sentence that needs to be classified
    Classify a sentence based on the intents 
    '''
    data = request.json
    sent = data.get('sent','')
    res = {}
    rank = {}
    if sent:
        res = aug_interpreter.parse(sent)
        rank['Intent'] = [res['intent_ranking'][i]['name'] for i in range(len(res['intent_ranking']))]
        rank['Confidence'] = [str(np.round(res['intent_ranking'][i]['confidence']*100,2))+' %' for i in range(len(res['intent_ranking']))]
    return rank
  
# main driver function
if __name__ == '__main__':
  
    # run() method of Flask class runs the application 
    # on the local development server.
    app.run()
