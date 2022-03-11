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
df = pd.read_excel('data/All_Intents.xlsx')

def classify_context(question, intent):
    if intent == 'Delivery':
        model_directory = 'models/delivery/'
        interpreter = Interpreter.load(model_directory)
        context = interpreter.parse(question)
        return context
    if intent == 'Subscription':
        model_directory = 'models/subscription/'
        interpreter = Interpreter.load(model_directory)
        context = interpreter.parse(question)
        return context
    if intent == 'General':
        model_directory = 'models/general/'
        interpreter = Interpreter.load(model_directory)
        context = interpreter.parse(question)
        return context
    if intent == 'Nutrition':
        model_directory = 'models/nutrition/'
        interpreter = Interpreter.load(model_directory)
        context = interpreter.parse(question)
        return context
    if intent == 'Support':
        model_directory = 'models/support/'
        interpreter = Interpreter.load(model_directory)
        context = interpreter.parse(question)
        return context

def get_similar_questions(question, context,intent):
    df1 = df[df.Intents == intent]
    context = context.get('name','')
    df1 = df1[df1.Contexts == context]
    question_list = df1.Questions.to_list()
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
    if intent == 'Subscription':
        bert_model_directory = 'outputs/subscription/'
        model = QuestionAnsweringModel('bert', bert_model_directory, use_cuda=False)
        to_predict = [{ "context": context, "qas": [{ "question": question, "id": 150 }] }]
        answers, probabilities = model.predict(to_predict)
        return str(answers[0]['answer'][0]),probabilities[0]['probability'][0]
    if intent == 'General':
        bert_model_directory = 'outputs/general/'
        model = QuestionAnsweringModel('bert', bert_model_directory, use_cuda=False)
        to_predict = [{ "context": context, "qas": [{ "question": question, "id": 150 }] }]
        answers, probabilities = model.predict(to_predict)
        return str(answers[0]['answer'][0]),probabilities[0]['probability'][0]
    if intent == 'Nutrition':
        bert_model_directory = 'outputs/nutrition_qa_model/best_model/'
        model = QuestionAnsweringModel('bert', bert_model_directory, use_cuda=False)
        to_predict = [{ "context": context, "qas": [{ "question": question, "id": 150 }] }]
        answers, probabilities = model.predict(to_predict)
        return str(answers[0]['answer'][0]),probabilities[0]['probability'][0]
    if intent == 'Support':
        bert_model_directory = 'outputs/support/'
        model = QuestionAnsweringModel('bert', bert_model_directory, use_cuda=False)
        to_predict = [{ "context": context, "qas": [{ "question": question, "id": 150 }] }]
        answers, probabilities = model.predict(to_predict)
        return str(answers[0]['answer'][0]),probabilities[0]['probability'][0]
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
    if confidence < 0.5:
        otherquestions = get_similar_questions(question, contextObj,intent)
        
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
