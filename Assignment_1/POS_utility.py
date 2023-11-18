import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import itertools
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Masking, Concatenate, Input


def exclude_symbols(s):
    return not all(i in string.punctuation for i in s)


def embedding_for_vocab(filepath, word_index, embedding_dim): 
    vocab_size = len(word_index) + 1
    embedding_matrix_vocab = np.zeros((vocab_size, embedding_dim)) 
    with open(filepath, encoding="utf8") as f: 
        for line in f: 
            word, *vector = line.split() 
            if word in word_index: 
                idx = word_index[word] 
                embedding_matrix_vocab[idx] = np.array( 
                    vector, dtype=np.float32)[:embedding_dim] 
    f.close()
    return embedding_matrix_vocab 
  
    
def build_base_model(LSTM_nodes, classes, model_name="nlp_model", additional_layers=[], return_seq=False):
    model = keras.Sequential(name=model_name)
    model.add(Input(shape=(100,1)))
    model.add(Bidirectional(LSTM(LSTM_nodes, return_sequences=return_seq)))
    for layer in additional_layers:
        model.add(layer)
    model.add(Dense(classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_seq_model(LSTM_nodes, classes, model_name="nlp_model", additional_layers=[], return_seq=True):
    model = keras.Sequential(name=model_name)
    model.add(Input(shape=(None,100)))
    model.add(Masking(mask_value=0.))
    from keras.layers import TimeDistributed
    # Ignores padding tokens
    model.add(Bidirectional(LSTM(LSTM_nodes, return_sequences=return_seq)))
    for layer in additional_layers:
        model.add(layer)
    model.add(TimeDistributed(Dense(classes, activation='softmax')))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def plot_averages(history):
    tmp = []
    tmp_val = []
    fig, ax = plt.subplots(1,2,figsize=(16,6))
    for h in history:
        tmp.append(h['accuracy'])
        tmp_val.append(h['val_accuracy'])
    ax[0].plot(np.mean(tmp, axis=0), color="#11aa00")
    ax[1].plot(np.mean(tmp_val, axis=0), color="#ffa520")
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
    plt.show()
    
    
def plot_single_runs(history):
    fig, ax = plt.subplots(1,len(history),figsize=(len(history)*6,5))
    for x in range(len(history)):
        ax[x].plot(history[x]['accuracy'])
        ax[x].plot(history[x]['val_accuracy'])
        ax[x].set_ylabel('accuracy')
        ax[x].set_xlabel('epoch')
        ax[x].legend(['train', 'val'], loc='upper left')
    plt.show()
    
    
def errors_summary(scores_dicts, encoder, train_y, test_y):
    low_scores = dict((x,y) for x,y in scores_dicts[0]["Scores"].items() if y < 0.5)
    low_classes = encoder.inverse_transform(list(low_scores.keys()))
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


def evaluate_model(model, checkpoint_path, test_X, test_y, metric, runs=1):
    scores = []
    for r in range(runs):
        model.load_weights(checkpoint_path.joinpath(str(r)))
        print("Testing model n."+str(r+1))
        y_pred = np.argmax(model.predict(test_X, verbose=2),axis=1)
        scores.append(metric(y_pred, test_y))
    return scores


def impermanent_training(model, ckp_path, Train_X, Train_Y, Val_X, Val_Y, batch_size=64, seeds=[]):
    """
    Input arguments: model: compiled Keras model, ckp_path: Path to a file
    
    Trains a model 3+ times restarting with fixed random seeds, and resets its weights afterwards.
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
            patience=2,
            verbose=1
        )

        history = model.fit(Train_X, Train_Y, batch_size=batch_size, epochs=10, validation_data=(Val_X, Val_Y),
                        callbacks=[checkpoint_callback, early_stop_callback], verbose=1)
        histories.append(history.history)
        model.set_weights(reset)
    return histories

def pad_sentences(sentences, embedding_dim, cat):
    """
    Function to convert a Pandas Series of uneven sentences into a Numpy array of fixed-size padded sentences.
    
    Inputs: sentences: Pandas Series, cat: string
    """
    from warnings import simplefilter
    simplefilter(action='ignore', category=FutureWarning)
    
    samples = 100
    if cat != 'train':
        samples = int(samples/2)
    
    X2 = sentences.to_numpy()
    X2 = np.column_stack((itertools.zip_longest(*X2, fillvalue=np.zeros(embedding_dim))))
    X2 = np.stack(X2).reshape(
        (samples, int(X2.shape[1]/embedding_dim), embedding_dim)).astype(float)
    print(f"Final Shape for {cat}:")
    print(X2.shape)
    return X2