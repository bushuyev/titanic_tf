import tensorflow as tf
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import tempfile
import os
import sklearn.metrics as sk_metrics

from logistic_regression_tf import Normalize, LogisticRegression
from plot_utils import confusion_matrix_plot, range_train_test_plot, corr_plot

matplotlib.rcParams['figure.figsize'] = [9, 6]
pd.set_option('display.max_rows', 500)

def prepare_data(df):

    df.set_index('PassengerId', inplace = True)
    df['Age'] = df['Age'].fillna(df['Age'].mean())


    sex_one_hot = pd.get_dummies(df['Sex'])

    df = df.join(sex_one_hot)

    df = df.drop(columns=["Name", "Sex", "Ticket", "Cabin", "Embarked"])

    return df

def main(data = "data/train.csv"):

    train_data= prepare_data(pd.read_csv(data))
    # test_data = prepare_data(pd.read_csv("data/test.csv"))

    corr_plot(train_data)

    tmp_train = train_data.sample(frac=0.75, random_state=1)
    test_data = train_data.drop(tmp_train.index)
    train_data = tmp_train

    print(train_data.info())

    print(train_data.head(10))

    x_train, y_train = train_data.iloc[:, 1:], train_data.iloc[:, 0].to_frame()
    x_test, y_test = test_data.iloc[:, 1:], test_data.iloc[:, 0].to_frame()


    print(x_train.head(10))
    print(y_train.head(10))

    print("------------------------------")
    print(type(x_train))
    print(type(y_train))
    print("+++++++++++++++++++++++++++++++")

    # print(x_test.head())
    # print(y_test.head())


    # print(train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))



    x_train, y_train = tf.convert_to_tensor(x_train, dtype=tf.float32), tf.convert_to_tensor(y_train, dtype=tf.float32)
    x_test, y_test = tf.convert_to_tensor(x_test, dtype=tf.float32), tf.convert_to_tensor(y_test, dtype=tf.float32)

    # sns.pairplot(train_data, hue = 'Survived', diag_kind='kde');
    print(train_data.describe().transpose()[:10])

    norm_x = Normalize(x_train)
    x_train_norm = norm_x.norm(x_train)
    x_test_norm =  norm_x.norm(x_test)

    log_reg = LogisticRegression()

    y_pred = log_reg(x_train_norm[:5], train=False)
    # print(y_pred.numpy())


    batch_size = 64

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_norm, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=x_train.shape[0]).batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test_norm, y_test))
    test_dataset = test_dataset.shuffle(buffer_size=x_test.shape[0]).batch(batch_size)


    # --------------------------------------------------------------------------------------
    # Set training parameters

    epochs = 200



    train_losses, test_losses, train_accs, test_accs = log_reg.train(epochs, train_dataset, test_dataset)
    # --------------------------------------------------------------------------------------
    range_train_test_plot(epochs, train_losses, test_losses, "loss")
    range_train_test_plot(epochs, train_accs, test_accs, "accuracy")


    y_pred_train, y_pred_test = log_reg(x_train_norm, train=False), log_reg(x_test_norm, train=False)
    train_classes, test_classes = log_reg.predict_class(y_pred_train), log_reg.predict_class(y_pred_test)
    confusion_matrix_plot(y_train, train_classes, 'Training')



if __name__ == "__main__":
    main()
#
# print(train_data.isnull().sum())
# print("----------------")
# print(train_data.groupby(["Name"]).filter(lambda x: len(x) > 0))
# print("----------------")
#
# plt.hist(train_data["Age"])
# plt.show()


# print(train_data['Name'].unique())

# train_data.pivot_table('Age',index=['FamilySize','Title'],aggfunc=['min','mean','median','max','count'])

# print(train_data)