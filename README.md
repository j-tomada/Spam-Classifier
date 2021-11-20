# Spam Classifier

This program will use the SKlearn library and Naive Bayes Classifier to determine whether or not an email is classified as spam. I have provided the testing and training set that I used for this project.


# Usage

The first step is build a vocabulary using the training set. Run **tr_vocab.py** which goes through each email and extracts all of the unique words stated through the entire training set. We are going to utilize this created dictionary as the features of the emails coming from the testing set. If an email has a similar frequency of words that are determined as spam, than we can make the assumption that the tested email is also spam.

Next, we run **Build_xy.py**. This does exactly what we mentioned earlier. We go through each email, both the testing and training set, and create the features by counting the frequency of words (from the created dictionary) that appear in each email. The program will then create two **.npy** files which holds the features of both the training and testing email. We will use .npy files to fit into the **Naive Bayes Classifier**

Finally, we run **Baiyes . py**. This will utilize the Naive Bayes Classifier from the SKlearn library. We first fit the tr_x.npy file as **X** and the **spam-mail.tr.label** as **y**. After fitting the testing data, we can now run our test on any email from the testing data set. We get the tt_x.npy file and we use that as a parameter for the classifiers' predict function. This will output a list of classifications (0 and 1) which determines whether the training email is considered spam.
