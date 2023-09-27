import string
import re
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import nltk
from sklearn.pipeline import make_pipeline
from lime_counterfactual.lime_counterfactual import LimeCounterfactual
from datasets import load_dataset


def clean_text(text):
    text = text.lower()
    text = ''.join([n for n in text if n != '\\'])
    text = ''.join([n for n in text if not n.isdigit()])
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if
            word.lower() not in nltk.corpus.stopwords.words('english') or
            word.lower() != "<br />"]
    return " ".join(text)


def remove_words(s: str, words: list):
    for word in words:
        s = re.sub(r"\b(%s)\b" % word, "?", s, flags=re.IGNORECASE)
    return s


if __name__ == "__main__":
    train_data = load_dataset("imdb", split="train")
    test_data = load_dataset("imdb", split="test")

    x_train = train_data["text"][9000:16000]
    y_train = train_data["label"][9000:16000]
    x_test = test_data["text"]
    y_test = test_data["label"]

    random.seed(42)
    random_ix_0 = random.sample(range(12500), 50)
    random_ix_1 = random.sample(range(12500, 25000), 50)
    random_ix = random_ix_0 + random_ix_1
    f = open("imdb_indices.txt", "w")
    f.write("negative:\n" + ','.join(str(i) for i in random_ix_0))
    f.write("\n\npositive:\n" + ','.join(str(i) for i in random_ix_1))
    f.close()
    x_test = [x_test[ix] for ix in random_ix]
    y_test = [y_test[ix] for ix in random_ix]

    x_train_clean = [clean_text(text) for text in x_train]
    x_test_clean = [clean_text(text) for text in x_test]

    tfidf_vectorizer = TfidfVectorizer()
    train_tfidf_matrix = tfidf_vectorizer.fit_transform(x_train_clean)
    test_tfidf_matrix = tfidf_vectorizer.transform(x_test_clean)

    feature_names = tfidf_vectorizer.get_feature_names_out()
    train_tfidf_array = train_tfidf_matrix.toarray()
    test_tfidf_array = test_tfidf_matrix.toarray()

    rf = RandomForestClassifier(n_estimators=100, random_state=1)
    rf.fit(train_tfidf_array, y_train)

    c = make_pipeline(tfidf_vectorizer, rf)
    class_fn = lambda x: rf.predict_proba(x)[:, 1]
    lime_c = LimeCounterfactual(c, class_fn, tfidf_vectorizer, 0.5, feature_names)

    data_counterfactual = pd.DataFrame()
    data_counterfactual["Review"] = ""
    data_counterfactual["Label"] = 0
    data_counterfactual["Predicted"] = 0
    data_counterfactual["Explanation size"] = 0
    data_counterfactual["Removed elements"] = ""
    data_counterfactual["Original score"] = 0
    data_counterfactual["New score"] = 0
    data_counterfactual["Counterfactual"] = ""
    data_positive_counterfactual = data_counterfactual.copy()
    data_negative_counterfactual = data_counterfactual.copy()

    empty_expl = 0
    wrong_class = 0
    for i, x_to_pred in enumerate(test_tfidf_array):
        y_pred = rf.predict([x_to_pred])[0]
        y_act = y_test[i]
        if y_pred != y_act:
            wrong_class += 1

        review_original = x_test[i]
        review = x_test_clean[i]

        if y_pred == 1:
            explanation = lime_c.explanation(review)
            lime_words = feature_names[explanation["explanation_set"]]
            if len(lime_words) == 0:
                empty_expl += 1
            inst = {
                "Review": review_original,
                "Label": y_act,
                "Predicted": y_pred,
                "Explanation size": explanation["size explanation"],
                "Removed elements": ''.join(e+"," for e in lime_words),
                "Original score": explanation["original score"],
                "New score": explanation["new score"],
                "Counterfactual": remove_words(review_original, lime_words),
            }
            data_counterfactual.loc[len(data_counterfactual)] = inst
            data_positive_counterfactual.loc[len(data_counterfactual)] = inst

        else:
            inst = {
                "Review": review_original,
                "Label": y_act,
                "Predicted": y_pred,
                "Explanation size": -1,
                "Removed elements": "/",
                "Original score": -1,
                "New score": -1,
                "Counterfactual": "/",
            }
            data_counterfactual.loc[len(data_counterfactual)] = inst
            data_negative_counterfactual.loc[len(data_counterfactual)] = inst

    data_counterfactual.to_csv("imdb_lime_c.csv", index=False)
    data_positive_counterfactual.to_csv("imdb_lime_c_good.csv", index=False)
    data_negative_counterfactual.to_csv("imdb_lime_c_bad.csv", index=False)

    ca = (len(y_test) - wrong_class) / len(y_test)
    print(f"ca: {ca}, empty cfe: {empty_expl}")
