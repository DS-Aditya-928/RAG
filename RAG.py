from openai import OpenAI
import numpy as np
import math

client = OpenAI(api_key="Your API key")


def cosine_simularity(A, B):
    return np.dot(A,B) / ( np.linalg.norm(A) * np.linalg.norm(B) )

content = " "

with open('C:\\Users\\Aditya.D.S\\Downloads\\messi.txt', 'r') as file:
    content = file.read()
    #print(content)

dictionary = {}
reqStr = []

for i in range(math.ceil(len(content)/50.0)):
    subStr = content[(i*50):((i*50)+50)]
    reqStr += [subStr]

t = client.embeddings.create(input= reqStr,
    model="text-embedding-3-small"
    )

print("Done!")

chunkLength = 100

for i in range(math.ceil(len(content)/float(chunkLength))):
    subStr = content[(i*chunkLength):((i*chunkLength)+chunkLength)]
    dictionary[subStr] = t.data[i].embedding
    #print(subStr)
    #print("\n")


while True:
    question = input('Ask a question about Messi\n')
    qEmbedding = client.embeddings.create(
    input=question,
    model="text-embedding-3-small"
    ).data[0].embedding

    newDict = {}

    for key in dictionary:
        #caalculate distance for each key(i.e data chunk) and the wuery.
        dist = cosine_simularity(qEmbedding, dictionary[key])
        newDict[dist] = key

    largest5 = sorted(newDict.keys())
    askString = question
    for i in range(len(largest5) - 1, len(largest5) - 6, -1):
        print(largest5[i])
        print(newDict[largest5[i]])
        askString += " " + newDict[largest5[i]]

    print(askString)#ask the llm with this query.
