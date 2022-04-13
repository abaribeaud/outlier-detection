import os
import pandas as pd
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from skmultilearn.problem_transform import ClassifierChain
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from spellchecker import SpellChecker


def __run__():
    """
        Main function
    """

    stopwords = open("stopwords.txt", 'r').read().split()

    class_knwon = pd.DataFrame()
    class_outlier = pd.DataFrame()

    for file in os.listdir("data_arthur"):
        df = pd.read_json("data_arthur/" + file, orient="record")
        class_knwon = pd.concat([class_knwon, df], ignore_index=True)

    for file in os.listdir("data_test_arthur"):
        df = pd.read_json("data_test_arthur/" + file, orient="record")
        class_outlier = pd.concat([class_outlier, df], ignore_index=True)

    test = class_knwon[1:10]
    print(test.columns)
    print(test["content"])
    test["content"] = test.apply(lambda x: rm_digit_and_spe_char(x["content"], stopwords), axis=1)
    print(test["content"])
    """
    vectorizer_bow = CountVectorizer(max_features=5000)
    vectorizer_tfidf = TfidfVectorizer()

    vectorizer.fit(class_knwon['content'])

    X_known = vectorizer.transform(class_knwon["content"])
    X_outlier = vectorizer.transform(class_outlier["content"])

    y_known = pd.get_dummies(class_knwon["category"])

    clf = ClassifierChain(
        classifier=SVC(kernel="linear"),
        require_dense=[False, True]
    )

    clf.fit(X_known, y_known)

    predictions = clf.predict(X_outlier)

    for i in predictions:
        print(i.toarray())
    """

def rm_digit_and_spe_char(text, stopwords):
    """
        Prepare and clean text :
            - Remove digit
            - Remove special character
            - Remove stopword
            - Correct typo with pyspellchecher

    :param text: text to clean
    :param stopwords: list of stopword used to remove stopword in text

    :return: cleaned text
    :rtype: str
    """

    spell = SpellChecker(language="fr", distance=2)  # fix distance to 1 for shorter run times

    text_output = " "
    for word in text.split():
        word = re.sub(r'\d+', "", word)  # remove digital char
        word = re.sub(r'[\@!-+°—"-_*()=,;:./?…|<>«»]', " ", word)  # remove special character
        word = word.lower()  # normalize to lower case

        # Check if the word is myspell
        word = "" if spell.unknown([word]) else word

        if word not in stopwords:
            text_output += " " + word

    return text_output


if __name__ == "__main__":
    __run__()
