from numpy import float64, isin
import pandas as pd
import nltk
from nltk.corpus import words

vocab = {}
data = pd.read_csv('TR.csv')
nltk.download('words')
set_words = set(words.words())

def buildVocab(currEmail):
    i = len(vocab)

    for word in currEmail:
        if word.lower() not in vocab and word.lower() in set_words:
            vocab[word] = i
            i = i + 1

if __name__ == '__main__':
    for i in range(data.shape[0]):
        currEmail = data.iloc[i,0].split()
        buildVocab(currEmail)

    print(f'Dictionary has been created ({len(data)} emails and length of {len(vocab)})')
    print("Writing dictionary onto vocab.txt")

    file = open("vocab.txt", "w")
    file.write(str(vocab))
    file.close
