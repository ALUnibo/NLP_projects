import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import itertools
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Masking, Concatenate, Input, TimeDistributed


def exclude_symbols(s):
    return not all(i in string.punctuation for i in s)

    
def build_base_model(LSTM_nodes, classes, model_name="nlp_model", additional_layers=[], return_seq=False):
    """
    Returns a baseline model, with the option to add more layers between the two default ones.
    
    Arguments: {LSTM_nodes: int, the number of nodes of the base LSTM;
                classes: int, the number of nodes of the classifier;
                model_name: str (optional), used as the model's name, defaults to 'nlp_model';
                additional_layers: list of keras.layers (optional), list of additional layers to be added to the model, defaults to [];
                return_seq: bool (optional), set this to True if the first additional layer needs sequences as inputs, defaults to False otherwise.}
                
    Returns: Compiled keras.Sequential model.
    """
    model = keras.Sequential(name=model_name)
    model.add(Input(shape=(100,1)))
    model.add(Bidirectional(LSTM(LSTM_nodes, return_sequences=return_seq)))
    for layer in additional_layers:
        model.add(layer)
    model.add(Dense(classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_seq_model(LSTM_nodes, classes, model_name="nlp_model", additional_layers=[]):
    """
    Returns a baseline model working with sentences instead of single words, with the option to add more layers between the two default ones.
    
    Arguments: {LSTM_nodes: int, the number of nodes of the base LSTM;
                classes: int, the number of nodes of the classifier;
                model_name: str (optional), used as the model's name, defaults to 'nlp_model';
                additional_layers: list of keras.layers (optional), list of additional layers to be added to the model, defaults to [];
                return_seq: bool (optional), set this to False if the first additional layer does not need sequences as inputs, defaults to True otherwise.}
                
    Returns: Compiled keras.Sequential model.
    """
    model = keras.Sequential(name=model_name)
    model.add(Input(shape=(None,100)))
    model.add(Masking(mask_value=0.))
    # Ignores padding tokens
    
    model.add(Bidirectional(LSTM(LSTM_nodes, return_sequences=True)))
    for layer in additional_layers:
        model.add(layer)
    model.add(TimeDistributed(Dense(classes, activation='softmax')))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def plot_averages(histories):
    """
    Plots two graphs of multiple training runs, one for Training accuracy and one for Validation accuracy, displaying the average as well as the single runs.
    
    Arguments: {histories: list of tf.keras.callbacks.History.history, one for each training run.}
    
    Returns: None
    """
    tmp = []
    tmp_val = []
    fig, ax = plt.subplots(1,2,figsize=(16,6))
    for h in histories:
        tmp.append(h['accuracy'])
        tmp_val.append(h['val_accuracy'])
    avg_tmp = np.mean(tmp, axis=0)
    avg_tmp_val = np.mean(tmp_val, axis=0)
    
    ax[0].plot(avg_tmp, color="#11aa00")
    ax[1].plot(avg_tmp_val, color="#ffa520")
    for h in tmp:
        ax[0].plot(h, color="#11aa0055")
    for h in tmp_val:
        ax[1].plot(h, color="#ffa52055")
    # plt.plot(np.mean(tmp_val, axis=0))
    ax[0].set_title("Training set")
    ax[1].set_title("Validation set")
    
    for idx in range(2):
        ax[idx].set_ylabel('avg_accuracy')
        ax[idx].set_xlabel('epoch')
        ax[idx].legend(['average', 'iterations'], loc='upper left')
    ax[0].text(len(avg_tmp)-0.5, avg_tmp[-1], str(avg_tmp[-1]*100)[0:4]+"%", fontsize=12, fontweight=600)
    ax[1].text(len(avg_tmp_val)-0.5, avg_tmp_val[-1], str(avg_tmp_val[-1]*100)[0:4]+"%", fontsize=12, fontweight=600)
    plt.show()
    
    
def plot_single_runs(histories):
    """
    Plots graphs of multiple training runs with both Training and Validation accuracy, displaying one graph per run.
    
    Arguments: {histories: list of tf.keras.callbacks.History.history, one for each training run.}
    
    Returns: None
    """
    fig, ax = plt.subplots(1,len(histories),figsize=(len(histories)*6,5))
    for x in range(len(histories)):
        ax[x].plot(histories[x]['accuracy'])
        ax[x].plot(histories[x]['val_accuracy'])
        ax[x].set_ylabel('accuracy')
        ax[x].set_xlabel('epoch')
        ax[x].legend(['train', 'val'], loc='upper left')
    plt.show()
    

def evaluate_model(model, checkpoint_path, test_X, test_y, metric, runs=1, sequences=False):
    scores = []
    for r in range(runs):
        model.load_weights(checkpoint_path.joinpath(str(r)))
        print("Testing model n."+str(r+1))
        y_scores = model.predict(test_X, verbose=2)
        y_pred = np.argmax(y_scores, axis=len(y_scores.shape)-1)
        scores.append(metric(y_pred, test_y, sequences=sequences))
    return scores
    
    
def errors_summary(scores_dicts, encoder, train_y, test_y):
    """
    Builds a pandas.DataFrame containing each class with a low (<0.5) F1 score, including how many times it appeared in both Training and Test set. 
    
    Arguments: {scores_dicts: list of dictionaries with required key "Scores", containing F1 scores for each class of the task (output of the 'evaluate_model' function);
                encoder: instance of sklearn.preprocessing.LabelEncoder used to encode the classes;
                train_y: numpy.array of int, Training labels used to train the model
                test_y: numpy.array of int, Test labels used to evaluate the model}
                
    Returns: pandas.Dataframe
    """
    low_scores = dict((x,y) for x,y in scores_dicts[0]["Scores"].items() if y < 0.75)
    low_classes = encoder.inverse_transform([int(i) for i in low_scores.keys()])
    train_y = encoder.inverse_transform(train_y)
    test_y = encoder.inverse_transform(test_y)
    low_train = pd.Series(train_y).value_counts().reindex(
        low_classes, fill_value=0)[low_classes]
    low_test = pd.Series(test_y).value_counts().reindex(
        low_classes, fill_value=0)[low_classes]
    low_df = pd.DataFrame(data=np.column_stack((low_test.index, low_train.values,
                                                low_test.values, list(low_scores.values()))),
                         columns=['POS class', 'train count', 'test count', 'score'])
    return low_df


def impermanent_training(model, ckp_path, Train_X, Train_Y, Val_X, Val_Y, seeds=[], **kwargs):
    """
    Trains a model any number of times and resets its weights afterwards, applying a random restart with given seeds. No seeds in input will only do 1 training cycle with a random integer as seed. The weights of the best models for each iteration will be saved to a file. 
    
    Arguments: {model: compiled Keras model, whose weights will not change at the end of execution;
                ckp_path: path to a file, to which a suffix will be added to distinguish different runs;
                Train_X, Train_Y, Val_X, Val_Y: input data for model.fit();
                batch_size: int (optional), defaults to 64;
                seeds: list of int (optional), each representing a different training run}
                
    Returns: list of tf.keras.callbacks.History.history
    """
    histories = []
    if not seeds:
        seeds = [random.randint()]
    for s in range(len(seeds)):
        reset = model.get_weights()
        print(f"Beginning training {s+1}/{len(seeds)}")
        random.seed(seeds[s])
        checkpoint_filepath = ckp_path.joinpath(str(s))
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min', 
            save_best_only=True)
        early_stop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            min_delta=0.001,
            patience=3,
            verbose=1
        )

        history = model.fit(Train_X, Train_Y, validation_data=(Val_X, Val_Y), verbose=1,
                        callbacks=[checkpoint_callback, early_stop_callback], **kwargs)
        histories.append(history.history)
        model.set_weights(reset)
    return histories

def pad_sentences(sentences, embedding_dim):
    """
    Converts a pandas.Series of uneven sentences into a Numpy array of fixed-size padded sentences.
    
    Arguments: {sentences: pandas.Series containing numpy.Array of uneven size, to be padded;
                embedding_dim: int, the length of an embedded word in the sentence}
                
    Returns: 3-dimensional numpy.Array
    """
    from warnings import simplefilter
    simplefilter(action='ignore', category=FutureWarning)
    
    samples = len(sentences)
    X2 = sentences.to_numpy()
    X2 = np.column_stack((list(itertools.zip_longest(*X2, fillvalue=np.zeros(embedding_dim)))))
    X2 = np.stack(X2).reshape(
        (samples, int(X2.shape[1]/embedding_dim), embedding_dim)).astype(float)
    print("Final Shape (sentences, sentence length, embedding dimensions):")
    print(X2.shape)
    return X2