import nltk

from lib.VO2FuncType_functions import sort_dict

lemmatizer = nltk.WordNetLemmatizer()


def nltk_tag_to_wordnet_tag(nltk_tag):
    from nltk.corpus import wordnet
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(sentence):
    # tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    # tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            # else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)


def eval_classifcation(model, dataset, apply_softmax=False, pairs=True):
    sent = [input("Enter Sentence:").lower()]
    if pairs:
        sent.append(input('Enter sentence 2:'))
    sent = [lemmatize_sentence(s) for s in sent]
    labels = dataset.lbl2type
    labels[len(labels)] = "Random"
    p = model.predict(sent, apply_softmax=apply_softmax).tolist()
    scores = sort_dict({labels[i]: round(v, 4) for i, v in enumerate(p)})
    for i, id2 in enumerate(scores):
        print("%s:\t%s" % (id2, str(scores[id2])), sep="\n")
        if i > 5:
            break