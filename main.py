import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report


def main():
    # the goal is to predict if the borrower will pay back the money or not

    # reading loan_data.csv as a dataframe
    loans = pd.read_csv('loan_data.csv')

    # checking out the loans information
    print(loans.info())
    print(loans.head())
    print(loans.describe())

    # histogram of two FICO distributions on top of each other, one for each credit.policy outcome
    plt.figure(figsize = (10, 6))
    loans[loans['credit.policy'] == 1]['fico'].hist(alpha = 0.5, bins = 30, color = 'blue', label = 'Credit policy = 1')
    loans[loans['credit.policy'] == 0]['fico'].hist(alpha = 0.5, bins = 30, color = 'red', label = 'Credit policy = 0')
    plt.legend()
    plt.xlabel('FICO')

    # similar figure, except this time selected by the not.fully.paid column
    plt.figure(figsize = (10, 6))
    loans[loans['not.fully.paid'] == 1]['fico'].hist(alpha = 0.5, bins = 30, label = 'Not fully paid = 1', color = 'blue')
    loans[loans['not.fully.paid'] == 0]['fico'].hist(alpha = 0.5, bins = 30, label = 'Not fully paid = 0', color = 'red')
    plt.legend()
    plt.xlabel('FICO')

    # countplot showing the counts of loans by purpose, with the color hue defined by not.fully.paid
    plt.figure(figsize = (15, 6))
    sns.countplot(data = loans, hue = 'not.fully.paid', x = 'purpose')

    # trend between FICO score and interest rate
    sns.jointplot(data = loans, kind = 'scatter', x = 'fico', y = 'int.rate')

    # lmplots to see if the trend differs between not.fully.paid and credit.policy
    sns.lmplot(x = 'fico', y = 'int.rate', data = loans, hue = 'credit.policy', col = 'not.fully.paid')

    # purpose column is categorical; transforming them using dummy variables
    cat_feats = ['purpose']
    final_data = pd.get_dummies(loans, columns = cat_feats, drop_first = True)
    print(final_data.info())

    # splitting data into a training set and a testing set
    X = final_data.drop('not.fully.paid', axis = 1)
    y = final_data['not.fully.paid']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

    # instance of DecisionTreeClassifier() and fitting it to the training data
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)

    # predictions from the test set and a classification report and a confusion matrix.
    predictions = dtree.predict(X_test)
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

    # instance of the RandomForestClassifier class and fitting it to the training data
    rf = RandomForestClassifier(n_estimators = 600)
    rf.fit(X_train, y_train)

    # predicting the class of not.fully.paid for the X_test data
    rf_predictions = rf.predict(X_test)

    # classification report and confusion matrix from the results
    print(classification_report(y_test, rf_predictions))
    print(confusion_matrix(y_test, rf_predictions))

    plt.show()


if __name__ == '__main__':
    main()