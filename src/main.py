import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf


from logistic_regression_tf import Normalize, LogisticRegression
from plot_utils import confusion_matrix_plot, range_train_test_plot, corr_plot


def cabin_to_deck(cabin):
    return cabin[0] if type(cabin) == str and len(cabin) > 0 else "X"


def prepare_data(df):

    print("""
            Minimal data preparation: 
                - age missing age replace with mean among class and sex
                - Cabin numbers replaced with decks (first letter) or X for missing values
                - decks and sex replaced with one hot columns
                - 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked' dropped as deemed insignificant
    """)

    df.set_index('PassengerId', inplace = True)

    age_by_class_and_sex = df.pivot_table('Age', index=['Pclass', 'Sex'], aggfunc=['min', 'mean', 'median', 'max', 'count'])

    df['Age'] = df.apply(
        lambda row: age_by_class_and_sex.loc[row['Pclass'], row['Sex']]['mean']['Age'] if np.isnan(row['Age']) else row['Age'],
        axis=1
    )

    df['Cabin'] = df.apply(
        lambda row: cabin_to_deck(row['Cabin']),
        axis=1
    )

    deck_one_hot = pd.get_dummies(df['Cabin'], prefix='Deck', dtype=float)
    df = df.join(deck_one_hot)

    print(deck_one_hot)

    sex_one_hot = pd.get_dummies(df['Sex'], prefix='Sex', dtype=float)
    df = df.join(sex_one_hot)

    # df = df.drop(columns=['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Deck_X', 'Deck_T', 'Deck_G', 'Deck_F', 'Deck_E', 'Deck_D', 'Deck_C', 'Deck_B', 'Deck_A'])
    df = df.drop(columns=['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Deck_T'])

    return df


def add_missing_dummy_values(df_full, df_to_fix, val):
    for c in set(df_full.columns).difference(df_to_fix.columns):
        df_to_fix[c] = val


def main(data = "data/train.csv"):

    raw_data = pd.read_csv(data)

    all_data= prepare_data(raw_data)

    row_test_data = pd.read_csv("data/test_.csv")
    row_test_data['Deck_T'] = 0.0
    test_data = prepare_data(row_test_data)

    add_missing_dummy_values(all_data, test_data, 0)

    print(test_data.head())

    corr_plot(all_data)

    train_data = all_data.sample(frac=0.75, random_state=1)
    valid_data = all_data.drop(train_data.index)

    x_train, y_train = train_data.iloc[:, 1:], train_data.iloc[:, 0].to_frame()
    x_valid, y_valid = valid_data.iloc[:, 1:], valid_data.iloc[:, 0].to_frame()
    x_test,  y_test = test_data.iloc[:, 1:], test_data.iloc[:, 0].to_frame()


    x_train, y_train = tf.convert_to_tensor(x_train, dtype=tf.float32), tf.convert_to_tensor(y_train, dtype=tf.float32)
    x_valid, y_valid = tf.convert_to_tensor(x_valid, dtype=tf.float32), tf.convert_to_tensor(y_valid, dtype=tf.float32)
    x_test,  y_test = tf.convert_to_tensor(x_test, dtype=tf.float32),  tf.convert_to_tensor(y_test, dtype=tf.float32)

    # sns.pairplot(train_data, hue = 'Survived', diag_kind='kde');
    print(train_data.describe().transpose()[:10])

    norm_x = Normalize(x_train)
    x_train_norm = norm_x.norm(x_train)
    x_valid_norm =  norm_x.norm(x_valid)
    x_test_norm =  norm_x.norm(x_test)

    log_reg = LogisticRegression()

    # y_pred = log_reg(x_train_norm[:5], train=False)
    # print(y_pred.numpy())


    batch_size = 64

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_norm, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=x_train.shape[0]).batch(batch_size)

    valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid_norm, y_valid))
    valid_dataset = valid_dataset.shuffle(buffer_size=x_valid.shape[0]).batch(batch_size)


    # --------------------------------------------------------------------------------------
    epochs = 200

    train_losses, valid_losses, train_accs, valid_accs = log_reg.train(epochs, train_dataset, valid_dataset)
    # --------------------------------------------------------------------------------------
    range_train_test_plot(epochs, train_losses, valid_losses, "loss")
    range_train_test_plot(epochs, train_accs, valid_accs, "accuracy")


    y_pred_train, y_pred_valid, y_pred_test = log_reg(x_train_norm, train=False), log_reg(x_valid_norm, train=False), log_reg(x_test_norm, train=False)
    train_classes, valid_classes, test_classes = log_reg.predict_class(y_pred_train), log_reg.predict_class(y_pred_valid), log_reg.predict_class(y_pred_test)
    # confusion_matrix_plot(y_train, train_classes, 'Training')
    confusion_matrix_plot(y_valid, valid_classes, 'Validation')
    confusion_matrix_plot(y_test, test_classes, 'Test')
    # print(test_classes)



if __name__ == "__main__":
    matplotlib.rcParams['figure.figsize'] = [9, 6]
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)

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