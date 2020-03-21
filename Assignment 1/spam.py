import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
# Styles
plt.style.use('ggplot')
sns.set_style('whitegrid')
# Text Preprocessing
import nltk
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize



dataset = pd.read_csv("spam.csv",encoding='latin-1')
dataset = dataset[["v1","v2"]]
dataset.head()



dataset = dataset.rename(columns={"v1":"label","v2":"text"})
dataset.head()



print(dataset.info())
print(dataset.label.value_counts())



dataset["label"].value_counts().plot(kind = 'pie', explode = [0, 0.1], figsize = (6, 6), autopct = '%1.1f%%', shadow = True)
plt.ylabel("Spam vs Ham")
plt.legend(["Ham", "Spam"])
plt.show()



topMessages = dataset.groupby("text")["label"].agg([len, np.max]).sort_values(by = "len", ascending = False).head(n = 10)
display(topMessages)
dataset.drop_duplicates(keep=False, inplace=True)



print(dataset.shape)
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer("english")



def cleanText(message):
    message = message.translate(str.maketrans('', '', string.punctuation))
    words = [stemmer.stem(word) for word in message.split() if word.lower() not in stopwords.words("english")]
    return " ".join(words)


dataset["text"] = dataset["text"].apply(cleanText)
dataset = dataset[["label","text"]]
dataset.head()

def train_test_split(dataset): 
    random_val_msk = np.random.rand(len(dataset)) <= 0.8
    X_train = dataset[random_val_msk]
    X_test = dataset[~random_val_msk]
    return X_train, X_test

dataset_train, dataset_test = train_test_split(dataset)
print(dataset_train.head())
print(dataset_test.head())

mask = np.random.rand(len(dataset))<0.8
text_train = dataset.text[mask].values
text_test = dataset.text[~mask].values
label_train = dataset.label[mask].values
label_test = dataset.label[~mask].values
print(len(text_train),len(text_test),len(label_train),len(label_test))


messages = dataset.text.values
words_all = []
for message in messages:
    words_all += (message.split(" "))
unique_words = set(words_all)
dictionary_words = {i:words_all.count(i) for i in unique_words}
dictionary_words['hello']



spam_messages = dataset.text.values[dataset.label == "spam"]
spam_words = []
for spam in spam_messages:
    spam_words += (spam.split(" "))
unique_spam_words = set(spam_words)
dictionary_spam = {i:spam_words.count(i) for i in unique_spam_words}
dictionary_spam['win']



ham_messages = dataset.text.values[dataset.label == "ham"]
ham_words = []
for ham in ham_messages:
    ham_words += (ham.split(" "))
unique_ham_words = set(ham_words)
dictionary_ham = {i:ham_words.count(i) for i in unique_ham_words}
dictionary_ham['love']



total_words = len(words_all)
total_spam = len(spam_words)
total_ham = len(ham_words)
print(total_words, total_spam, total_ham)



def probability_word_given_spam(word):
    return (dictionary_spam[word]/total_spam) 
def probability_word_given_ham(word):
    return dictionary_ham[word]/total_ham 
def probability_word(word):
    try:
        return dictionary_words[word]/total_words
    except KeyError:
        return 0.000000001 
def probability_of_message_being_spam(message):
    num = den = 1
    for word in message.split():
        if word in spam_words:
            num *= probability_word_given_spam(word)
            den *= probability_word(word)
    # This step ensures laplace smoothing 
    if den==0:
        num+=1
        den+=1
    return num/den
def probability_of_message_being_ham(message): 
    num = den = 1
    for word in message.split():
        if word in ham_words:
            num *= probability_word_given_ham(word)
            den *= probability_word(word)
    if den==0:
        num+=1
        den+=1
    return num/den
def spam_predictor(mess):
    if probability_of_message_being_spam(mess) >= probability_of_message_being_ham(mess):
        return "spam"
    else:
        return "ham"




def accuracy_prediction(text_test, label_test):
    false_positive = false_negative = 0 
    true_positive = true_negative = 0
    for i,m in enumerate(text_test):
        predicted = spam_predictor(m)
        actual = label_test[i]
        if predicted == "spam" and actual == "spam":
            true_negative+=1
        if predicted == "spam" and actual == "ham":
            false_negative+=1
        if predicted == "ham" and actual == "spam":
            false_positive+=1
        if predicted == "ham" and actual == "ham":
            true_positive+=1
    accuracy = (true_negative+true_positive)/len(text_test)
    return accuracy, false_positive, false_negative, true_positive, true_negative




acc,fp,fn,tp,tn = accuracy_prediction(text_test,label_test)
print(acc*100, fp,fn,tp,tn)



print("True positive: ",tp,"\n False positive: ",fp,"\n False Negative: ",fn,"\n True Negative: ",tn)