import pandas as pd
import numpy as np
import ast

'''
Program builds the x and y set for both training and testing data
Essentially goes through the created dictionary (using Spam_vocab.py)
and finds the total count of each words used in an email
The frequency of words is used to determine whether somethng is considered
spam or not
Name: Joseph Tomada
Date: 09/30/2021
'''
if __name__ == '__main__':

    file = input("TR or TT (Build training or Testing?): ")

    if file == 'TR' or file == 'TT':
        data = pd.read_csv(str(file)+'.csv')
        file = open('vocab.txt', 'r')
        contents = file.read()
        vocab = ast.literal_eval(contents)

        x = np.zeros((data.shape[0], len(vocab)))

        for i in range(data.shape[0]):
            email = data.iloc[i, 0].split()

            for emailWord in email:
                if emailWord.lower() in vocab:
                    x[i, vocab[emailWord.lower()]] += 1
        np.save('tr_x.npy', x)


        if file == 'TR':
            data = pd.read_csv('spam-mail.tr.label')
            y = data['Prediction'].values
            np.save('tr_y.npy', y)
    else:
        print('Must input TR or TT')
