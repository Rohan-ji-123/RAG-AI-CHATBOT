import requests
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed",json={
        "model" : "bge-m3",
        "input" : text_list
        },
        timeout=300 
    )
    r.raise_for_status()                 # fail loudly on HTTP errors
    embedding = r.json()['embeddings']
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate",json={
        "model":"llama3.2",
        "prompt":prompt,
        "stream":False
    })
    response = r.json()
    print(response)
    return response 

df = joblib.load('embeddings.joblib')    

incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]
# print(question_embedding)

similarities = cosine_similarity(np.vstack(df['embedding']),[question_embedding]).flatten()
# print(similarities)
top_result = 3
max_indx = similarities.argsort()[::-1][0:top_result]
print(max_indx)
new_df = df.loc[max_indx]
# print(new_df[['title','number','text']])

# for index, item in new_df.iterrows():
#     print(index,item["title"],item["number"],item["text"])

prompt = f''' I am teaching web development in my Sigma web development course. Here are video
subtitle chunk containig video title, video number, start time in seconds, end time in
second , the text at that time: 
{new_df[["title","number","start","end","text"]].to_json(orient="records")}
-------------------------------------------------------------------------------------
"{incoming_query}"
user asked this question related to the video chunk , you have to answer in a human way (dont mention
the above format , its just for you ) where and how much content is taught in which video (in which video and at what timesmap) and guide the user to go
to that particular video. If video asks unrelated question, tell him that you can olny answer question
related to the course
'''
with open("promt.txt", "w")as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)

with open("response.txt","w") as f:
    f.write(response)