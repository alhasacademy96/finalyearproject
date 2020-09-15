#   Author: Ibrahim Alhas - ID: 1533204.

#   MODEL 2:    GloVe Embeddings with pre-trained vectors (included in folder).
#   This is the final version of the model (not the base).

#   Packages and libraries used for this model.
#   ** Install these if not installed already **.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from keras.callbacks import TensorBoard
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
import nltk
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from bs4 import BeautifulSoup
import re, string, unicodedata
from tensorflow.keras.preprocessing import text, sequence  # tensorflow
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_curve
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from string import punctuation
from nltk import pos_tag
from nltk.corpus import wordnet
from tensorflow import keras as tff
# import keras
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf

#   Importing dataset. Note that the dataset is divided into true (factual) and fake (fictitious) subsets.
#   ** Make sure that the directory is correct, otherwise it will give a "no such file or directory" error **.
true = pd.read_csv("True.csv")
false = pd.read_csv("Fake.csv")

#   Basic data visualisation -------------------------------------------------------------------------------------------
#   We see that the title column is from news articles, and the text column forms the twitter tweet extracts.
print(true.head())
print(false.head())

#   We set the labels for each data instance, where factual = 1, otherwise 0.
true['category'] = 1
false['category'] = 0

#   We merge the two divided datasets (true and fake) into a singular dataset.
df = pd.concat([true, false])

#   We can see that the dataset is a bit unbalanced, with more instances for fiction.
print(sns.set_style("darkgrid"))
print(sns.countplot(df.category))

#   Checking for missing values (i.e. Nan).
print(df.isna().sum())
df.title.count()
df.subject.value_counts()

#   We merge the columns title and text into one column.
#   We also delete the columns subject, date and title.
df['text'] = df['text'] + " " + df['title']
del df['title']
del df['subject']
del df['date']

#   We use STOPWORDS for this model.
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
print(stop.update(punctuation))

#   DATA CLEANING-------------------------------------------------------------------------------------------------------
#   We incorporate the publishers feature from title and text instances, and place it into the dataset manually.
#   First Creating list of index that do not have publication part. We can use this as a new feature.
unknown_publishers = []
for index, row in enumerate(true.text.values):
    try:
        record = row.split(" -", maxsplit=1)
        # if no text part is present, following will give error
        print(record[1])
        # if len of piblication part is greater than 260
        # following will give error, ensuring no text having "-" in between is counted
        assert (len(record[0]) < 260)
    except:
        unknown_publishers.append(index)

#   We print the instances where publication information is absent or different.
print(true.iloc[unknown_publishers].text)

#   We want to use the publication information as a new feature.
publisher = []
tmp_text = []
for index, row in enumerate(true.text.values):
    if index in unknown_publishers:
        #   Append unknown publisher:
        tmp_text.append(row)
        publisher.append("Unknown")
        continue
    record = row.split(" -", maxsplit=1)
    publisher.append(record[0])
    tmp_text.append(record[1])

#   Replace text column with new text + add a new feature column called publisher/source.
true["publisher"] = publisher
true["text"] = tmp_text
del publisher, tmp_text, record, unknown_publishers
# -
#   Validate that the publisher/source column has been added to the dataset.
print(true.head())

#   Check for missing values, then drop them for both datasets.
print([index for index, text in enumerate(true.text.values) if str(text).strip() == ''])
true = true.drop(8970, axis=0)
fakeEmptyIndex = [index for index, text in enumerate(false.text.values) if str(text).strip() == '']
print(f"No of empty rows: {len(fakeEmptyIndex)}")
false.iloc[fakeEmptyIndex].tail()


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


# Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


# Removing URL's
def remove_between_square_brackets(text):
    return re.sub(r'http\S+', '', text)


# Removing the stopwords from text
def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)


#   Remove noise for better performance.
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_stopwords(text)
    return text


#   We apply the denoise function on our column in the dataset.
df['text'] = df['text'].apply(denoise_text)

#   Text distribution is different for the classes: 2500 characters is most common in true category,
#   ...while around 5000 characters in common in fake text category.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
text_len = df[df['category'] == 1]['text'].str.len()
ax1.hist(text_len, color='red')
ax1.set_title('Original text')
text_len = df[df['category'] == 0]['text'].str.len()
ax2.hist(text_len, color='blue')
ax2.set_title('Fake text')
fig.suptitle('Characters in texts')
plt.show()

#   N words in each class.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
text_len = df[df['category'] == 1]['text'].str.split().map(lambda x: len(x))
ax1.hist(text_len, color='green')
ax1.set_title('Original text')
text_len = df[df['category'] == 0]['text'].str.split().map(lambda x: len(x))
ax2.hist(text_len, color='black')
ax2.set_title('Fake text')
fig.suptitle('Words in texts')
plt.show()

#   Average words for each label.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
word = df[df['category'] == 1]['text'].str.split().apply(lambda x: [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)), ax=ax1, color='blue')
ax1.set_title('Original')
word = df[df['category'] == 0]['text'].str.split().apply(lambda x: [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)), ax=ax2, color='black')
ax2.set_title('Fake')
fig.suptitle('Average word length for each text')


def get_corpus(text):
    words = []
    for i in text:
        for j in i.split():
            words.append(j.strip())
    return words


#   Print the first 5 in the dataset.
corpus = get_corpus(df.text)
print(corpus[:5])

#   Find the most common words (upto 10).
counter = Counter(corpus)
mostCommonWords = counter.most_common(10)
mostCommonWords = dict(mostCommonWords)
print(mostCommonWords)


def get_top_text_ngrams(corpus, n, g):
    vec = CountVectorizer(ngram_range=(g, g)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]


#   We split the data into training and splitting, as presented in the report.
#   We found that for this model, the train/test split ratio was: 2:3.
x_train, x_test, y_train, y_test = train_test_split(df.text, df.category, random_state=0)

#   Parameters for our upcoming model-----------------------------------------------------------------------------------
maxFeatures = 10000
maxLength = 300

#   We tokenize the words (representing every word with a vectorized number).
tokenizer = text.Tokenizer(num_words=maxFeatures)
tokenizer.fit_on_texts(x_train)
tokenized_train = tokenizer.texts_to_sequences(x_train)
x_train = sequence.pad_sequences(tokenized_train, maxlen=maxLength)
tokenized_test = tokenizer.texts_to_sequences(x_test)
X_test = sequence.pad_sequences(tokenized_test, maxlen=maxLength)

#   For the embedding, we use GloVe. We believe that using this format will produce in better performances.
#   We use a pre-trained file to forward into our model, for better performance.

#   Tutorial for glove embeddings: https://nlp.stanford.edu/pubs/glove.pdf
EMBEDDING_FILE = 'glove.twitter.27B.200d.txt'


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


#   Embedding: https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db
embedIndexing = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding="utf8"))
# embedIndexing = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding="utf8"))

embeddingsAll = np.stack(embedIndexing.values())
emb_mean, emb_std = embeddingsAll.mean(), embeddingsAll.std()
sizeOfEmbed = embeddingsAll.shape[1]

word_index = tokenizer.word_index
nb_words = min(maxFeatures, len(word_index))
# change below line if computing normal stats is too slow
embedding_matrix = embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, sizeOfEmbed))
for word, i in word_index.items():
    if i >= maxFeatures: continue
    embedding_vector = embedIndexing.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

#   Model parameters that were fine-tuned.
batch_size = 256
epochs = 5
sizeOfEmbed = 200
log_dir = "logs\\model\\"
#   A custom callbacks function, which initially included tensorboard.
mycallbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),  # Restoring the best
    #   ...weights will help keep the optimal weights.
    #   tf.keras.callbacks.TensorBoard(log_dir="./logs"),  # NEWLY ADDED - CHECK.
    tf.keras.callbacks.TensorBoard(log_dir=log_dir.format(time())),  # NEWLY ADDED - CHECK.
    #   tensorboard --logdir logs --> to check tensorboard feedback.
]

#   We define the neural network----------------------------------------------------------------------------------------
model = Sequential()
#   Embedding layer (which is un-trainable).
model.add(
    Embedding(maxFeatures, output_dim=sizeOfEmbed, weights=[embedding_matrix], input_length=maxLength, trainable=False))
model.add(LSTM(units=128, return_sequences=True, recurrent_dropout=0.25, dropout=0.25))
model.add(LSTM(units=64, recurrent_dropout=0.1, dropout=0.1))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

print("Model weights:")
print(model.weights)

#   Track training history, for visualisations.
history = model.fit(x_train, y_train, batch_size=batch_size, validation_data=(X_test, y_test), epochs=epochs,
                    callbacks=mycallbacks)

#   We evaluate our model by predicting a few instances from our test data (the first 5)--------------------------------
print("Stats:")
print("Accuracy of train data:", model.evaluate(x_train, y_train)[1] * 100, "%")
print("Accuracy of test data:", model.evaluate(X_test, y_test)[1] * 100, "%")

#   Produce a figure, for every epoch, and show performance metrics.
epochs = [i for i in range(5)]
fig, ax = plt.subplots(1, 2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(20, 10)

ax[0].plot(epochs, train_acc, 'go-', label='Training Accuracy')
ax[0].plot(epochs, val_acc, 'ro-', label='Testing Accuracy')
ax[0].set_title('Training & Testing Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs, train_loss, 'go-', label='Training Loss')
ax[1].plot(epochs, val_loss, 'ro-', label='Testing Loss')
ax[1].set_title('Training & Testing Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
plt.show()

#   We predict a few instances (up to 5). For all instances with probability 0.5 and over, it is fiction; else factual.
print("Predictions:")
pred = model.predict_classes(X_test)
print(pred[:5])

#   We print the performance metrics.
print("Report:")
print(classification_report(y_test, pred, target_names=['Fact', 'Fiction']))

#   We print the confusion matrix.
print("C Matrix:")
cm = confusion_matrix(y_test, pred)
print(cm)

print("Ibrahim Alhas")

cmm = pd.DataFrame(cm, index=['Fake', 'Original'], columns=['Fake', 'Original'])
plt.figure(figsize=(10, 10))
sns.heatmap(cm, cmap="Blues", linecolor='black', linewidth=1, annot=True, fmt='', xticklabels=['Fake', 'Original'],
            yticklabels=['Fake', 'Original'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#   End----------------------------------------------------
# print('Accuracy on testing set:', accuracy_score(binary_predictions, y_test))
# print('Precision on testing set:', precision_score(binary_predictions, y_test))
# print('Recall on testing set:', recall_score(binary_predictions, y_test))
# print('F1 on testing set:', f1_score(binary_predictions, y_val))

#   End----------------------------------------------------
