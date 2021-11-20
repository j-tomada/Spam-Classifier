import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

def split_input_data():
    x = np.load('tr_x.npy')
    y = np.load('tr_y.npy')
    X_train, X_test, y_train, y_test = train_test_split(x, y)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    input = int(input('Evaluate a split training set or testing set?\n(1.)Split (2.)TT\n'))
    x_train, x_test, y_train, y_test = 0, 0, 0, 0
    if input == 1:
        x_train, x_test, y_train, y_test = split_input_data()
    elif input == 2:
        x_train = np.load('tr_x.npy')
        y_train = np.load('tr_y.npy')
        x_test = np.load('tt_x.npy')
    else:
        print("Wrong input")
        exit()

    NB = MultinomialNB()
    NB.fit(x_train, y_train)
    y_pred = NB.predict(x_test)

    if input == 1:
        print('accuracy: ' +str(sum(y_pred == y_test)/x_test.shape[0]))
    elif input == 2:
        print("Writing results onto results.csv")
        '''
        Writes classification results onto results.csv
        '''
        with open('results.csv', 'w', newline='') as csvfile: 
            csvwriter = csv.writer(csvfile)
            count = 1
            for prediction in y_pred:
                csvwriter.writerow([count, prediction])
                count += 1