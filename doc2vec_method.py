from cProfile import run
import os
from pydoc import doc
import pandas as pd
import re
import warnings

warnings.filterwarnings("ignore")
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from spellchecker import SpellChecker

def __run__():
    """
        Main function
    """

    stopwords = open("stopwords.txt", 'r').read().split()

    class_knwon = pd.read_csv("clean.csv")    
    class_outlier = pd.read_csv("outlier.csv")


    test = class_knwon[1:100]
   
    documents = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(test["content"])]

    m = Doc2Vec(documents= documents, dm = 1, vector_size=300, window=8, min_count=10, workers=4)

    #Enregistrer le modèle
    m.save("model/doc2vec.model")
    


def exploit():
    stopwords = open("stopwords.txt", 'r').read().split()

    class_outlier = pd.DataFrame()

    for file in os.listdir("data_test_arthur"):
        df = pd.read_json("data_test_arthur/" + file, orient="record")
        class_outlier = pd.concat([class_outlier, df], ignore_index=True)

    test = class_outlier[:1]
    test["content"] = test.apply(lambda x: rm_digit_and_spe_char(x["content"], stopwords), axis=1)


    m = Doc2Vec.load('model/doc2vec.model')
    vector = m.infer_vector(test["content"].str.split().tolist()[0])
    # print(m.similarity_unseen_docs(m, vector))



if __name__ == "__main__":
    __run__()
