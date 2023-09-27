import string
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import nltk
from sklearn.pipeline import make_pipeline
from lime_counterfactual.lime_counterfactual import LimeCounterfactual


def text_process(text):
    text = text.lower()
    text = ''.join([n for n in text if not n.isdigit()])
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in
            nltk.corpus.stopwords.words('english')]
    return " ".join(text)


def remove_words(s: str, words: list):
    for word in words:
        s = re.sub(r"\b(%s)\b" % word, "*", s, flags=re.IGNORECASE)
    return s


if __name__ == "__main__":
    data = pd.read_csv("spam_data.csv")
    data["Category"] = data["Category"].replace(['ham', 'spam'], [0, 1])
    class_names = ["ok", "spam"]

    X_train, X_test, y_train, y_test = train_test_split(data["Message"], data["Category"],
                                                        test_size=0.2, random_state=42)

    X_train_clean = X_train.apply(text_process)
    X_test_clean = X_test.apply(text_process)

    tfidf_vectorizer = TfidfVectorizer()
    train_tfidf_matrix = tfidf_vectorizer.fit_transform(X_train_clean)
    test_tfidf_matrix = tfidf_vectorizer.transform(X_test_clean)

    feature_names = tfidf_vectorizer.get_feature_names_out()
    train_tfidf_array = train_tfidf_matrix.toarray()
    test_tfidf_array = test_tfidf_matrix.toarray()

    y_test_arr = y_test.to_numpy()
    x_test_clean_arr = X_test_clean.to_numpy()
    x_test_original_arr = X_test.to_numpy()

    rf = RandomForestClassifier(n_estimators=100, random_state=1)
    rf.fit(train_tfidf_array, y_train)

    c = make_pipeline(tfidf_vectorizer, rf)
    class_fn = lambda x: rf.predict_proba(x)[:, 1]
    lime_c = LimeCounterfactual(c, class_fn, tfidf_vectorizer, 0.5, feature_names)

    data_counterfactual = pd.DataFrame()
    data_counterfactual["Message"] = ""
    data_counterfactual["Label"] = 0
    data_counterfactual["Predicted"] = 0
    data_counterfactual["Explanation size"] = 0
    data_counterfactual["Removed elements"] = ""
    data_counterfactual["Original score"] = 0
    data_counterfactual["New score"] = 0
    data_counterfactual["Counterfactual"] = ""
    data_spam_counterfactual = data_counterfactual.copy()
    data_ham_counterfactual = data_counterfactual.copy()

    empty_expl = 0
    wrong_class = 0
    for i, x_to_pred in enumerate(test_tfidf_array):
        y_pred = rf.predict([x_to_pred])[0]
        y_act = y_test_arr[i]
        if y_pred != y_act:
            wrong_class += 1

        message_original = x_test_original_arr[i]
        message = x_test_clean_arr[i]

        if y_pred == 1:
            explanation = lime_c.explanation(message)
            lime_words = feature_names[explanation["explanation_set"]]
            if len(lime_words) == 0:
                empty_expl += 1
            inst = {
                "Message": message_original,
                "Label": y_act,
                "Predicted": y_pred,
                "Explanation size": explanation["size explanation"],
                "Removed elements": ''.join(e+"," for e in lime_words),
                "Original score": explanation["original score"],
                "New score": explanation["new score"],
                "Counterfactual": remove_words(message_original, lime_words),
            }
            data_counterfactual.loc[len(data_counterfactual)] = inst
            data_spam_counterfactual.loc[len(data_counterfactual)] = inst

        else:
            inst = {
                "Message": message_original,
                "Label": y_act,
                "Predicted": y_pred,
                "Explanation size": -1,
                "Removed elements": "/",
                "Original score": -1,
                "New score": -1,
                "Counterfactual": "/",
            }
            data_counterfactual.loc[len(data_counterfactual)] = inst
            data_ham_counterfactual.loc[len(data_counterfactual)] = inst

    data_counterfactual.to_csv("spam_lime_c.csv", index=False)
    data_ham_counterfactual.to_csv("spam_lime_c_ham.csv", index=False)
    data_spam_counterfactual.to_csv("spam_lime_c_spam.csv", index=False)

    ca = (len(y_test_arr) - wrong_class) / len(y_test_arr)
    print(f"ca: {ca}, empty cfe: {empty_expl}")
