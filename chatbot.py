import numpy as np
import joblib
from keras.models import model_from_json

def load_model(model_filename, model_weights_filename):
    with open(model_filename, 'r', encoding='utf8') as f:
        model = model_from_json(f.read())
    model.load_weights(model_weights_filename)
    return model

def pad_seq(seq,symbol,maxlen):
    while len(seq) < maxlen:
        seq.append(symbol)
    return seq

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

def chat():
    tokenizer = joblib.load('/home/paulo/projetos/Seq2seq/tokenizer.pkl')
    index_to_word = {v:k for k,v in tokenizer.word_index.items()}
    encoder_model = load_model('encoder_model.json', 'encoder_model_weights.h5')
    decoder_model = load_model('decoder_model.json', 'decoder_model_weights.h5')
    print("Hello, this is a test script for a chatbot.")
    while True:
        msg = input()
        sent = pad_seq(tokenizer.texts_to_sequences([msg])[0],
                       8001,
                       40)
        sent = np.array(sent).reshape(1,-1)
        print(inference(encoder_model,
                        decoder_model,
                        sent,
                        8001,
                        8002,
                        index_to_word))

if __name__ == "__main__":
    chat()
