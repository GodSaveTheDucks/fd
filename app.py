from flask import Flask, request, jsonify
from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config
from rasa_nlu.model import Trainer, Metadata, Interpreter
from simpletransformers.question_answering import QuestionAnsweringModel

import rasa_nlu
import spacy
import numpy as np
import pandas as pd


app = Flask(__name__)
nlp = spacy.load('en')
bert_model_directory = 'outputs/delivery/bert/best_model/'
model = QuestionAnsweringModel('bert', bert_model_directory, use_cuda=False)
df = pd.read_excel('data/Delivery_Contexts4.xlsx', sheet_name='Data v3.2', nrows= 106, usecols=[0,1,2])


def classify_context(question, intent):
    if intent == 'Delivery':
        model_directory = 'models/base_context/'
        interpreter = Interpreter.load(model_directory)
        context = interpreter.parse(question)
        return context

def get_similar_questions(question, context):
    df1 = df[df.Context == context]
    question_list = df1.Question.to_list()
    q1_doc = nlp(question)
    question_doc = [nlp(que) for que in question_list]
    similar_ques = [q1_doc.similarity(doc) for doc in question_doc]
    similar_ques_df = pd.DataFrame(similar_ques, columns=['sim_val'])
    top_questions = similar_ques_df.sort_values('sim_val', ascending=False)[:5]
    return [question_list[i] for i in top_questions.index]


def qa_model(intent,context,question):
    if intent == 'Delivery':
        bert_model_directory = 'outputs/delivery/bert/best_model/'
        model = QuestionAnsweringModel('bert', bert_model_directory, use_cuda=False)

        to_predict = [{ "context": context, "qas": [{ "question": question, "id": 150 }] }]
        answers, probabilities = model.predict(to_predict)
        return str(answers[0]['answer'][0]),probabilities[0]['probability'][0]
        #return {}
    return ""
    
@app.route('/classifyContext',methods=['POST'])
def classifyContext():
    answer = None
    otherquestions = []
    confidence = None
    data = request.json
    question = data.get('question','')
    intent = data.get('intent','Delivery')
    context = classify_context(question,intent)
    contextObj = context['intent']
    context = context['intent']['name']
    answer,confidence = qa_model('Delivery', context, question)
    if confidence < 0.4:
        otherquestions = get_similar_questions(question, contextObj)
    return {
        "question" : question, 
        "context" : context,
        "answer" : answer,
        "other_question" : otherquestions,
        "confidence" : confidence}
  
    
# main driver function
if __name__ == '__main__':
  
    # run() method of Flask class runs the application 
    # on the local development server.
    app.run(debug=True)
