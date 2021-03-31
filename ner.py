# IMPORT LIBRARIES

import numpy as np
import pandas as pd
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle

# GLOBAL CONSTANTS
DATA_FILEPATH = 'ner_dataset.csv'
SAVED_MODEL_PATH = 'ner_rnn.h5'
ENCODING = "latin1"
MAX_LEN = 104
BATCH_SIZE=32 
EPOCHS=10 
TEST_SPLIT=0.2
VALIDATION_SPLIT=0.2

# FUNCTIONS

def preprocess_data(filepath, encoding):
    """
    Takes dataset, imputes missing values, generates a list of list of sentences 
    in which each token has (word, tag) pair.
    This function would be different for each given dataset
    """
    df = pd.read_csv(filepath, encoding=encoding)
    df.head()
    df['Sentence #'] = df['Sentence #'].fillna(method='ffill')
    #print('Number of missing values:')
    #print(df.isnull().sum())
    #print()
    
    sentences = df.groupby("Sentence #")["Word"].apply(list).values
    tags = df.groupby("Sentence #")["Tag"].apply(list).values
    
    paired_sentences = []
    for i in range(len(sentences)):
        sent_pair = list(zip(sentences[i],tags[i]))
        paired_sentences.append(sent_pair)
    
    return df, paired_sentences

def create_vocabulary(df):
    """
    Takes dataframe and returns both unique words and NER's, as well as their size.
    This function would be different for each given dataset
    """
    unique_words = df["Word"].unique().tolist()
    unique_words.append("ENDPAD")
    unique_words.append("UNK")
    num_words = len(unique_words)

    unique_tags = df["Tag"].unique().tolist()
    num_tags = len(unique_tags)
    
    return unique_words, unique_tags, num_words, num_tags


def create_mapping(unique_words, unique_tags):
    """
    Takes unique word + NER lists and encodes each item to an integer
    Returns the generated dictionaries
    """
    word2index = {w: i for i, w in enumerate(unique_words)}
    tag2index = {t: i for i, t in enumerate(unique_tags)}
    
    return word2index, tag2index

def extract_features(sentences, word2index, tag2index, num_tags):
    """
    Generates features and labels using the dictionaries.
    Applies padding to each word-vector (to ensure same size for each)
    Converts labels to one-hot vector
    Returns Train-Test split
    """
    X = [[word2index[w[0]] for w in s] for s in sentences]
    y = [[tag2index[w[1]] for w in s] for s in sentences]
    
    X = pad_sequences(maxlen=MAX_LEN, sequences=X, padding="post",value=word2index["ENDPAD"])
    y = pad_sequences(maxlen=MAX_LEN, sequences=y, padding="post", value=tag2index["O"])
    
    y = [to_categorical(i, num_classes=num_tags) for i in y]
    
    return train_test_split(X, y, test_size=TEST_SPLIT)

def create_model(num_words, num_tags):
    """
    Simple Bi-LSTM Recurrent Neural Network Architecture
    """
    model = Sequential()
    model.add(Embedding(input_dim=num_words, output_dim=MAX_LEN, input_length=MAX_LEN))
    model.add(Dropout(0.25))
    model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Dense(num_tags, activation="softmax")))

    return model

def fit_model(model, X_train, y_train):
    """
    Runs the model
    """
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    earlyStop = EarlyStopping(monitor='val_loss', patience=2)
    checkpointer = ModelCheckpoint(filepath=SAVED_MODEL_PATH, save_best_only=True)

    history = model.fit(X_train, np.array(y_train), 
                        batch_size=BATCH_SIZE, 
                        epochs=EPOCHS, 
                        validation_split=VALIDATION_SPLIT, 
                        callbacks=[earlyStop, checkpointer])

    return history

def inverse_mapping(tag2index):
    """
    Gets indexed dict and returns tagged dict (inverse-transform)
    """
    index2tag = {i: t for t, i in tag2index.items()}

    return index2tag

def evaluate_model(model, X_test, y_test, index2tag):
    """
    Makes predictions
    Chooses class with maximum probability
    Converts each indices back to tags
    Prints evaluation metrics and classification report
    """
    y_pred = model.predict(X_test)

    y_pred = np.argmax(y_pred, axis=-1)
    y_true = np.argmax(y_test, axis=-1)

    y_pred = [[index2tag[y] for y in y_s] for y_s in y_pred]
    y_true = [[index2tag[y] for y in y_s] for y_s in y_true]

    print("Accuracy score : {:.1%}".format(accuracy_score(y_true, y_pred)))
    print("Precision score: {:.1%}".format(precision_score(y_true, y_pred)))
    print("Recall score   : {:.1%}".format(recall_score(y_true, y_pred)))
    print("F1-score       : {:.1%}".format(f1_score(y_true, y_pred)))
    print()
    print(classification_report(y_true, y_pred))
    print()


def train_from_scratch():
    """
    Executes the functions above to train the model and save the results to disk
    """
    df, sentences = preprocess_data(DATA_FILEPATH, ENCODING)
    unique_words, unique_tags, num_words, num_tags = create_vocabulary(df)
    word2index, tag2index = create_mapping(unique_words, unique_tags)
    X_train, X_test, y_train, y_test = extract_features(sentences, word2index, tag2index, num_tags)
    model = create_model(num_words, num_tags)
    fit_model(model, X_train, y_train)
    index2tag = inverse_mapping(tag2index)
    evaluate_model(model, X_test, y_test, index2tag)

    obj = {'word2index': word2index, 'tag2index': tag2index, 'index2tag': index2tag, 
           'num_tags': num_tags, 'unique_words': unique_words}
    with open("saved_objects.dat", "wb") as f:
        pickle.dump(obj, f)


def use_pretrained_model(sentence, tags):
    """
    Loads the pre-trained model and saved pickle file
    # Converts each word and label into index
    # Applies padding and one-hot vectorization
    # Makes prediction and get true labels
    # Shows predictions
    """
    model = load_model(SAVED_MODEL_PATH)

    with open("saved_objects.dat", "rb") as f:
        saved_obj = pickle.load(f)
    word2index = saved_obj['word2index'] 
    tag2index = saved_obj['tag2index']  
    index2tag = saved_obj['index2tag']  
    num_tags = saved_obj['num_tags']  
    unique_words = saved_obj['unique_words']
    
    X = [word2index.get(word, word2index['UNK']) for word in sentence]
    y = [tag2index.get(tag, tag2index["O"]) for tag in tags]

    X.extend([word2index["ENDPAD"]] * (MAX_LEN - len(X)))
    y.extend([tag2index["O"]] * (MAX_LEN - len(y)))
    y = [to_categorical(i, num_classes=num_tags) for i in y]

    pred = model.predict(np.array([X]))
    pred = np.squeeze(np.argmax(pred, axis=-1))
    true = np.argmax(np.array(y), axis=-1)

    print("{:12}| {:6}| {}".format("Word", "True", "Pred"))
    print(30 * "-")
    for w, t, p in zip(np.array(X), true, pred):
        if unique_words[w] != 'ENDPAD':
            print("{:12}: {:6} {}".format(unique_words[w], index2tag[t], index2tag[p]))


         
########################################################################
# IF YOU WANT TO TRAIN THE MODEL FROM SCRATCH AND THEN MAKE A PREDICTION
########################################################################

train_from_scratch()

########################################################################
# IF YOU WANT TO MAKE PREDICTION USING PRE-TRAINED AND SAVED MODEL
# Comment the above "train_from_scratch()"" method 
# and run "use_pretrained_model()"" below
########################################################################
      
sentence = ['Mr.', 'Huseyin', 'said', 'the', 'latest', 'figures', 'show', '1.8', 'million', 'people', 
            'are', 'in', 'need', 'of', 'food', 'assistance', '-', 'with', 'the', 'need', 'greatest', 
            'in', 'Indonesia', ',', 'the', 'Maldives', 'and', 'India', '.']

tags = ['B-per', 'I-per', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
        'O', 'O', 'O', 'O', 'B-geo', 'O', 'O', 'B-geo', 'O', 'B-geo', 'O']

use_pretrained_model(sentence, tags)

