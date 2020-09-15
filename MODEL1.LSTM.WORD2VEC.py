#   Author: Ibrahim Alhas - ID: 1533204.

#   MODEL 1:    Word2vec Embeddings (self-trained).
#   This is the final version of the model (not the base).

#   Packages and libraries used for this model.
#   ** Install these if not installed already **.
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from time import time
# plt.style.use('ggplot')
import seaborn as sns
import nltk
import re
import tensorboard as board
import nltk
import gensim
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, confusion_matrix, \
    f1_score, roc_curve

#   We use STOPWORDS from NLTK package.
warnings.filterwarnings('ignore')
nltk.download('stopwords')
nltk.download('punkt')

#   Basic conData visualisation ----------------------------------------------------------------------------------------
false = pd.read_csv("Fake.csv")
print(false.head())
plt.figure(figsize=(8, 5))
sns.countplot("subject", data=false)
plt.show()

true = pd.read_csv("True.csv")
print(true.head())
sns.countplot("subject", data=true)
plt.show()

#   Cleaning the conData------------------------------------------------------------------------------------------------
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

#   Validate that the publisher/source column has been added to the dataset.
print(true.head())

#   Check for missing values, then drop them for both datasets.
print([index for index, text in enumerate(true.text.values) if str(text).strip() == ''])
true = true.drop(8970, axis=0)
fakeEmptyIndex = [index for index, text in enumerate(false.text.values) if str(text).strip() == '']
print(f"No of empty rows: {len(fakeEmptyIndex)}")
false.iloc[fakeEmptyIndex].tail()

# Also noticed false news have a lot of CPATIAL-CASES. Could preserve Cases of letters, but as we are using Google's
# pretrained word2vec vectors later on, which haswell-formed lower cases word. We will contert to lower case.
# The text for these rows seems to be present in title itself. Lets merge title and text to solve these cases.
# Looking at publication Information


#   Basic visualisation of conData, i.e. subjects such as politics.
for key, count in true.subject.value_counts().iteritems():
    print(f"{key}:\t{count}")
sns.countplot(x="subject", data=true)
plt.show()

# Pre-processing--------------------------------------------------------------------------------------------------------

#   We set the labels for each conData instance, where factual = 1, otherwise 0.
true["class"] = 1
false["class"] = 0

#   Combining title with text columns.
true["text"] = true["title"] + " " + true["text"]
false["text"] = false["title"] + " " + false["text"]

#   Because subjects are not the same for both datasets, we have to drop them to avoid bias.
true = true.drop(["subject", "date", "title", "publisher"], axis=1)
false = false.drop(["subject", "date", "title"], axis=1)

# Combining both datasets.
conData = true.append(false, ignore_index=True)
del true, false

# Download stopwords and punkt.
nltk.download('stopwords')
nltk.download('punkt')

# Removing STOPWORDS, punctuations, and single-characters---------------------------------------------------------------
y = conData["class"].values

# Converting input to acceptable format for gensim.
X = []
stop_words = set(nltk.corpus.stopwords.words("english"))
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
for par in conData["text"].values:
    tmp = []
    sentences = nltk.sent_tokenize(par)
    for sent in sentences:
        sent = sent.lower()
        tokens = tokenizer.tokenize(sent)
        filtered_words = [w.strip() for w in tokens if w not in stop_words and len(w) > 1]
        tmp.extend(filtered_words)
    X.append(tmp)
del conData

#   Vectorization using Word2vec----------------------------------------------------------------------------------------
#   We set the dimensions of the words with the following parameter:
embedDimensions = 100
epochs = 5

#   The function that converts the words into vectors.
w2v_model = gensim.models.Word2Vec(sentences=X, size=embedDimensions, window=5, min_count=1)

#   Print current vocabulary size generated via vectorization.
print(len(w2v_model.wv.vocab))

#   We pass these vectors into the LSTM model as integers instead of words.
#   Keras is useful for embedding words.

#   Since we cant pass words into the embedding layer directly, we have to tokenize the words.
#   Tokenization:-------------------------------------------------------------------------------------------------------
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)

#   We check the first 10 instances of vectors to validate they have been converted to vectors.
print(X[0][:10])

# Our word mappings are in the dictionary.
indexingWords = tokenizer.word_index
for word, num in indexingWords.items():
    print(f"{word} -> {num}")
    if num == 10:
        break

#   The variant of RNN we use here is many-to-one (as stated in the report). This is because we have multiple inputs,
#   ...but only a probability of factual or fictitious for each input.

#   We keep words that are 700 words in length, and truncate anything above 700.
maxLength = 700

#   Apply the max length of words here:
X = pad_sequences(X, maxlen=maxLength)

#   Embedding Layer creates 1 more vector for unknown, padded words. This Vector is sparse with 0s.
#   Thus our vocab size increases by 1.
vocabularySize = len(tokenizer.word_index) + 1


#   We define a function that creates our weight matrix for our neural network.
def get_weight_matrix(model, vocab):
    # Vocabulary size + 1
    vocab_size = len(vocab) + 1
    # Definition of  weight matrix dimensions with zeros.
    weight_matrix = np.zeros((vocab_size, embedDimensions))
    #   Vocabulary stepping, where we store vectors using tokenization number mapping
    for word, i in vocab.items():
        weight_matrix[i] = model[word]
    return weight_matrix


# We create a matrix of mapping between word-index and vectors. We use this as weights in embedding layer
# Embedding layer accepts numecical-token of word and outputs corresponding vercor to inner layer.
# It sends vector of zeros to next layer for unknown words which would be tokenized to 0.
# Input length of Embedding Layer is the length of each news (700 now due to padding and truncating)

# Getting embedding vectors from word2vec and usings it as weights of non-trainable keras embedding layer
embedVectors = get_weight_matrix(w2v_model, indexingWords)

#   A custom callbacks function, which initially included tensorboard.
mycallbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),  # Restoring the best
    #   ...weights will help keep the optimal weights.
]

# Defining Neural Network
model = Sequential()
model.add(
    Embedding(vocabularySize, output_dim=embedDimensions, weights=[embedVectors], input_length=maxLength,
              trainable=False))
# LSTM
model.add(LSTM(units=128))
model.add(Dense(1, activation='sigmoid'))
# model.add(Dropout(0.5))
#   Compiling our model here with hyperparameters, such as loss function.
#   We implemented various optimizers but adam was the best one.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
del embedVectors

model.summary()

print("Model weights:")
print(model.weights)

# Train test split ratio.
X_train, X_test, y_train, y_test = train_test_split(X, y)

#   We fit and train the model, with the hyperparameters dictated below, i.e. epochs.
history = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=256, shuffle=True,
                    callbacks=mycallbacks)

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

#   We evaluate our model.----------------------------------------------------------------------------------------------
print("Evaluation:")
print(model.evaluate(X_test, y_test))

#   We predict a few instances (up to 5). For all instances with probability 0.5 and over, it is fiction; else factual.
pred = (model.predict(X_test) >= 0.5).astype("int")
print(pred[:5])

binaryPred = []
for i in pred:
    if i >= 0.5:
        binaryPred.append(1)
    else:
        binaryPred.append(0)

#   We print performance metrics.
print('Accuracy on test set:', accuracy_score(binaryPred, y_test))
print('Precision on test set:', precision_score(binaryPred, y_test))
print('Recall on test set:', recall_score(binaryPred, y_test))
print('F1 on test set:', f1_score(binaryPred, y_test))

#   We print the confusion matrix.
print("Report:")
print(classification_report(y_test, pred, target_names=['Fact', 'Fiction']))

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
