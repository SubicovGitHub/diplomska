import pandas as pd
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


def polyjuice_prompt(text, cc, blanked_text, do_print=False):
    prompt = text + " <|perturb|> [" + cc + "] " + blanked_text
    if do_print:
        print(prompt)
    return prompt


# izmed vseh vrnjenih CFE-jev vrni unikatne
def unique_cfes(res):
    unique = []
    for r in res:
        r2 = r["generated_text"]
        r_split = r2.split("[SEP] ")[-1].split(" [")[0]
        if r_split not in unique:
            unique.append(r_split)
    return unique


def polyjuice_pipeline(gen, model, prem, hypo, og_pred, n=2):
    cc_list = ["negation", "quantifier", "shuffle", "lexical",
               "resemantic", "insert", "delete", "restructure"]

    all_cfes = []
    valid_cfes = []
    valid_cfe_labels = []
    text_words = hypo.split()
    for ix, word in enumerate(text_words):
        blanked_text = hypo.replace(word, "[BLANK]")
        for cc in cc_list:
            prompt = polyjuice_prompt(hypo, cc, blanked_text)
            try:
                res = gen(prompt, num_beams=2, num_return_sequences=2)
                unique_blanks = unique_cfes(res)
                for unique in unique_blanks:
                    cfe = blanked_text.replace("[BLANK]", unique)
                    # print(cfe)
                    all_cfes.append(cfe)
                    new_prediction = process_prediction(model(prem + " " + cfe))
                    # print(new_prediction)
                    if new_prediction != og_pred:
                        valid_cfes.append(cfe)
                        valid_cfe_labels.append(new_prediction)
                        if len(valid_cfes) >= n:
                            return all_cfes, valid_cfes, valid_cfe_labels
            except:
                print("error")
    return all_cfes, valid_cfes, valid_cfe_labels


def process_prediction(r):
    r_word = r[0]["label"]
    if r_word == "entailment":
        return 0
    elif r_word == "neutral":
        return 1
    else:
        return 2


if __name__ == "__main__":
    test_data = load_dataset("esnli", split="test")
    x_test_premise = test_data[0:99]["premise"]
    x_test_hypothesis = test_data[0:99]["hypothesis"]
    y_test = test_data[0:99]["label"]

    data_counterfactual = pd.DataFrame()
    data_counterfactual["Premise"] = ""
    data_counterfactual["Hypothesis"] = ""
    data_counterfactual["Label"] = 0
    data_counterfactual["Predicted"] = 0
    data_counterfactual["Counterfactual_1"] = ""
    data_counterfactual["Counterfactual_1_label"] = 0
    data_counterfactual["Counterfactual_2"] = ""
    data_counterfactual["Counterfactual_2_label"] = 0

    model_path = "uw-hai/polyjuice"
    generator = pipeline("text-generation",
                         model=AutoModelForCausalLM.from_pretrained(model_path),
                         tokenizer=AutoTokenizer.from_pretrained(model_path),
                         framework="pt", device=-1)
    classifier = pipeline("text-classification",
                          model="geckos/bart-fined-tuned-on-entailment-classification")

    wrong_class = 0
    for i, premise in enumerate(x_test_premise):
        hypothesis = x_test_hypothesis[i]

        model_input = premise + " " + hypothesis
        original_prediction = process_prediction(classifier(model_input))

        if original_prediction != y_test[i]:
            wrong_class += 1

        all_cf, valid_cf, valid_cf_labels = polyjuice_pipeline(generator, classifier, premise,
                                                               hypothesis, original_prediction)
        print(i)
        print(model_input)
        print(valid_cf)
        print(valid_cf_labels)
        print("---------------------")

        inst = {
            "Premise": premise,
            "Hypothesis": hypothesis,
            "Label": y_test[i],
            "Predicted": original_prediction,
            "Counterfactual_1": "" if len(valid_cf) < 1 else valid_cf[0],
            "Counterfactual_1_label": 3 if len(valid_cf_labels) < 1 else valid_cf_labels[0],
            "Counterfactual_2": "" if len(valid_cf) < 2 else valid_cf[1],
            "Counterfactual_2_label": 3 if len(valid_cf_labels) < 2 else valid_cf_labels[1],
        }
        data_counterfactual.loc[len(data_counterfactual)] = inst

    data_counterfactual.to_csv("esnli_polyjuice.csv", index=False, sep='|')

    ca = (len(y_test) - wrong_class) / len(y_test)
    print(f"ca: {ca}*")
