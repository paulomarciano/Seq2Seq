import pandas as pd
import numpy as np
import keras.layers as kl
import keras.models as km
import keras
import keras.backend as K
from math import ceil
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import joblib

def get_model(input_length,
              vocab_size,
              encoder_units,
              decoder_units,
              word_index,
              dropout_prob=.5,
              optimizer=keras.optimizers.RMSprop(lr=1e-3)):
    """
    returns a keras sequence to sequence model.
    input_length: length of the input sequence.
    vocab_size: size of the vocabulary, including <EOS> and unknown symbols.
    encoder_units: iterator with the number of units in each hidden (B)LSTM
    layer for the encoder.
    decoder_units: int that represents the number of units in decoder LSTM.
    It has to be equal to the last encoder layer.
    dropout_prob: dropout probability added between layers. defaults to .5.
    optimizer: optimizer to use for training. defaults to RMSprop

    return: a couple of compiled keras models, one for inference and the other
    for training.
    """
    embedding_dim = 100
    #Loading the GloVe representation.
    embeddings_index = {}
    f = open('./Cornell/glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # Building the encoder.
    inputs = kl.Input(shape=[input_length])
    embedding = kl.Embedding(len(word_index) + 1,
                             embedding_dim,
                             weights=[embedding_matrix],
                             trainable=False)
    embedding_inputs = embedding(inputs)
    last_layer = embedding_inputs
    encoder_depth = len(encoder_units)
    # iterate through units, layer by layer.
    for i,units in enumerate(encoder_units):
        # if final layer, stop returning sequences and start returning state.
        if i == encoder_depth - 1:
            encoder = kl.LSTM(units, return_state=True)
        # otherwise, just return full sequences.
        else:
            encoder = kl.Bidirectional(
                                       kl.LSTM(units, return_sequences=True)
                                      )(last_layer)
            dropout = kl.Dropout(dropout_prob)(encoder)
            last_layer = dropout

    # Discard output, keep the states.
    encoder_outputs, state_h, state_c = encoder(last_layer)

    # Building the decoder.
    decoder_inputs = kl.Input(shape=[None,vocab_size])
    # embedding_decoder = embedding(decoder_inputs)
    embedding_decoder = decoder_inputs
    decoder = kl.LSTM(decoder_units,
                      return_sequences=True,
                      return_state=True)
    decoder_out,_,_ = decoder (embedding_decoder,
                               initial_state=[state_h,state_c])
    decoder_projection = kl.Dense(embedding_dim,activation='relu')
    decoder_softmax = kl.Dense(vocab_size, activation='softmax')
    decoder_out = decoder_softmax(decoder_projection(decoder_out))

    # Making the models.
    model = km.Model([inputs,decoder_inputs], decoder_out)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    encoder_model = km.Model(inputs,[state_h,state_c])
    encoder_model.summary()

    decoder_states_inputs = [kl.Input(shape=[decoder_units]),
                             kl.Input(shape=[decoder_units])]
    decoder_outputs, state_h, state_c = decoder(embedding_decoder,
                                                initial_state=decoder_states_inputs)
    decoder_states = [state_h,state_c]
    decoder_outputs = decoder_softmax(decoder_projection(decoder_outputs))
    decoder_model = km.Model([decoder_inputs]+decoder_states_inputs,
                             [decoder_outputs]+decoder_states)

    return [model, encoder_model, decoder_model]

def inference(encoder,decoder,tokens,eos,vocab_size,word_index):
    states = encoder.predict(tokens)
    target_sequence = np.zeros((1, 1, vocab_size))
    target_sequence[0,0,eos] = 1
    stop_condition = False
    decoded = ''
    size = 0
    while not stop_condition:
        output, h, c = decoder.predict([target_sequence]+states)
        infered_token = np.argmax(output[0,-1,:])
        if infered_token == eos:
            infered_word = '<EOS>'
        elif infered_token ==0:
            infered_word = '<UNK>'
        else:
            infered_word = word_index[infered_token]
        decoded += infered_word+' '
        size += 1

        if infered_token == eos or size == tokens.shape[1]:
            stop_condition = True

        target_sequence = np.zeros((1, 1, vocab_size))
        target_sequence[0,0,infered_token] = 1

        states = [h,c]

    return decoded

def pad_seq(seq,symbol,maxlen):
    if len(seq)>maxlen:
        rseq = seq[:maxlen-1]
        rseq.append(symbol)
        return rseq
    while len(seq) < maxlen:
        seq.append(symbol)
    return seq

def batch_generator(targets,batch_size,num,vocab_size):
    return to_categorical(targets[num*batch_size:(num+1)*batch_size],
                          num_classes=vocab_size)

def load_model(model_filename, model_weights_filename):
    with open(model_filename, 'r', encoding='utf8') as f:
        model = model_from_json(f.read())
    model.load_weights(model_weights_filename)
    return model

def chat(tokenizer,
         encoder_model,
         decoder_model,
         vocab_size
        ):
    print("Hello, this is a test script for a chatbot.")
    while True:
        msg = input()
        sent = pad_seq(tokenizer.texts_to_sequences([msg])[0],
                       vocab_size - 1,
                       40)
        sent = np.array(sent).reshape(1,-1)
        print(inference(encoder_model,
                        decoder_model,
                        sent,
                        vocab_size - 1,
                        vocab_size,
                        index_to_word))

if __name__ == '__main__':
    # Start by getting the data ready for training.

    train_enc = open('./Cornell/train.enc','r')
    train_dec = open('./Cornell/train.dec','r')
    train_enc_list = []
    train_dec_list = []

    for line in train_enc:
        train_enc_list.append(line.lower())
    for line in train_dec:
        train_dec_list.append(line.lower())

    train_enc.close()
    train_dec.close()

    vocab_size = 10002
    tokenizer = Tokenizer(num_words=vocab_size-2,oov_token='<UNK>')
    tokenizer.fit_on_texts(np.hstack([train_enc_list,train_dec_list]))
    enc_seq = tokenizer.texts_to_sequences(train_enc_list)
    dec_seq = tokenizer.texts_to_sequences(train_dec_list)

    for i in range(len(enc_seq)):
        for j in range(len(enc_seq[i])):
            if enc_seq[i][j] > vocab_size - 1:
                enc_seq[i][j] = 0
        for j in range(len(dec_seq[i])):
            if dec_seq[i][j] > vocab_size - 1:
                dec_seq[i][j] = 0
        enc_seq[i]=pad_seq(enc_seq[i],vocab_size-1,40)
        dec_seq[i]=pad_seq(dec_seq[i],vocab_size-1,40)

    dec_target=[i.copy() for i in dec_seq]

    for i in range(len(dec_seq)):
        dec_seq[i].pop()
        dec_seq[i].insert(0,vocab_size-1)

    enc_seq=np.array(enc_seq)
    dec_seq=np.array(list(dec_seq))

    model, encoder_model, decoder_model = get_model(input_length=40,
                                                    vocab_size=vocab_size,
                                                    encoder_units=[512,256],
                                                    decoder_units=256,
                                                    word_index=tokenizer.word_index,
                                                    optimizer='rmsprop')

    # Actually start training
    model.summary()

    # You should change these.
    batch_size = 64
    epochs = 100

    s = 0
    k = 0

    index_to_word = {v:k for k,v in tokenizer.word_index.items()}
    for _ in tqdm(range(epochs)):
        for i in range(ceil(enc_seq.shape[0]/batch_size)):
            categorical_targets = batch_generator(dec_target,
                                                  batch_size,
                                                  i,
                                                  vocab_size)
            categorical_inputs = batch_generator(dec_seq,
                                                 batch_size,
                                                 i,
                                                 vocab_size)
            s += model.train_on_batch([enc_seq[i*batch_size:(i+1)*batch_size],
                                       categorical_inputs],
                                       # dec_seq[i*batch_size:(i+1)*batch_size]],
                                      categorical_targets)
            k += 1
        print(s/k)
        s=0
        k=0

    # save models so we can replicate later (Keras seems to have an issue with this.):
    model.save('encoder_decoder.h5')

    # If only life were so simple... Keras can't save these this easily yet.
    # encoder_model.save('encoder.h5')
    # decoder_model.save('decoder.h5')

    # Here's a workaround:
    with open('encoder_model.json', 'w', encoding='utf8') as f:
        f.write(encoder_model.to_json())
    encoder_model.save_weights('encoder_model_weights.h5')

    with open('decoder_model.json', 'w', encoding='utf8') as f:
        f.write(decoder_model.to_json())
    decoder_model.save_weights('decoder_model_weights.h5')

    joblib.dump(tokenizer,'tokenizer.pkl')

    # Test the model:
    chat(tokenizer,
         encoder_model,
         decoder_model,
         vocab_size
        )

    # When you need to load the models:
    # model = km.load_model('./encoder_decoder.h5')
    # encoder_model = load_model('encoder_model.json', 'encoder_model_weights.h5')
    # decoder_model = load_model('decoder_model.json', 'decoder_model_weights.h5')
    # tokenizer = joblib.load('./tokenizer.pkl')
