import tensorflow as tf
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import tempfile
import os
import sklearn.metrics as sk_metrics

def show_confusion_matrix(y, y_classes, typ):
    # Compute the confusion matrix and normalize it
    plt.figure(figsize=(10,10))
    confusion = sk_metrics.confusion_matrix(y.numpy(), y_classes.numpy())
    confusion_normalized = confusion / confusion.sum(axis=1, keepdims=True)
    axis_labels = range(2)
    ax = sns.heatmap(
        confusion_normalized, xticklabels=axis_labels, yticklabels=axis_labels,
        cmap='Blues', annot=True, fmt='.4f', square=True)
    plt.title(f"Confusion matrix: {typ}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

class Normalize(tf.Module):
    def __init__(self, x):
        # Initialize the mean and standard deviation for normalization
        self.mean = tf.Variable(tf.math.reduce_mean(x, axis=0), name="Mean")
        self.std = tf.Variable(tf.math.reduce_std(x, axis=0), name="Std")

    def norm(self, x):
        # Normalize the input
        return (x - self.mean)/self.std

    def unnorm(self, x):
        # Unnormalize the input
        return (x * self.std) + self.mean


class LogisticRegression(tf.Module):

    def __init__(self):
        self.built = False

    def __call__(self, x, train=True):
        # Initialize the model parameters on the first call
        if not self.built:
            # Randomly generate the weights and the bias term
            rand_w = tf.random.uniform(shape=[x.shape[-1], 1], seed=22)
            rand_b = tf.random.uniform(shape=[], seed=22)
            self.w = tf.Variable(rand_w, name="W")
            self.b = tf.Variable(rand_b, name="B")
            self.built = True
        # Compute the model output
        z = tf.add(tf.matmul(x, self.w), self.b)
        # z = tf.squeeze(z, axis=1)
        if train:
            return z
        return tf.sigmoid(z)

def prepare_data(df):

    df.set_index('PassengerId', inplace = True)
    df['Age'] = df['Age'].fillna(df['Age'].mean())



    sex_one_hot = pd.get_dummies(df['Sex'])

    df = df.join(sex_one_hot)

    df = df.drop(columns=["Name", "Sex", "Ticket", "Cabin", "Embarked"])

    return df

def predict_class(y_pred, thresh=0.5):
    # Return a tensor with  `1` if `y_pred` > `0.5`, and `0` otherwise
    return tf.cast(y_pred > thresh, tf.float32)

def accuracy(y_pred, y):
    # Return the proportion of matches between `y_pred` and `y`
    y_pred = tf.math.sigmoid(y_pred)
    y_pred_class = predict_class(y_pred)
    check_equal = tf.cast(y_pred_class == y,tf.float32)
    acc_val = tf.reduce_mean(check_equal)
    return acc_val

def log_loss(y_pred, y):
    # Compute the log loss function
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_pred)
    return tf.reduce_mean(ce)

# Preset matplotlib figure sizes.
matplotlib.rcParams['figure.figsize'] = [9, 6]
pd.set_option('display.max_rows', 500)



train_data= prepare_data(pd.read_csv("data/train.csv"))
# test_data = prepare_data(pd.read_csv("data/test.csv"))

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
learning_rate = 0.01
train_losses, test_losses = [], []
train_accs, test_accs = [], []

# Set up the training loop and begin training
for epoch in range(epochs):
    batch_losses_train, batch_accs_train = [], []
    batch_losses_test, batch_accs_test = [], []

    # Iterate over the training data
    for x_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            y_pred_batch = log_reg(x_batch)
            batch_loss = log_loss(y_pred_batch, y_batch)
        batch_acc = accuracy(y_pred_batch, y_batch)
        # Update the parameters with respect to the gradient calculations
        grads = tape.gradient(batch_loss, log_reg.variables)
        for g,v in zip(grads, log_reg.variables):
            v.assign_sub(learning_rate * g)
        # Keep track of batch-level training performance
        batch_losses_train.append(batch_loss)
        batch_accs_train.append(batch_acc)

    # Iterate over the testing data
    for x_batch, y_batch in test_dataset:
        y_pred_batch = log_reg(x_batch)
        batch_loss = log_loss(y_pred_batch, y_batch)
        batch_acc = accuracy(y_pred_batch, y_batch)
        # Keep track of batch-level testing performance
        batch_losses_test.append(batch_loss)
        batch_accs_test.append(batch_acc)

    # Keep track of epoch-level model performance
    train_loss, train_acc = tf.reduce_mean(batch_losses_train), tf.reduce_mean(batch_accs_train)
    test_loss, test_acc = tf.reduce_mean(batch_losses_test), tf.reduce_mean(batch_accs_test)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    if epoch % 20 == 0:
        print(f"Epoch: {epoch}, Training log loss: {train_loss:.3f}")
# --------------------------------------------------------------------------------------
plt.plot(range(epochs), train_losses, label = "Training loss")
plt.plot(range(epochs), test_losses, label = "Testing loss")
plt.xlabel("Epoch")
plt.ylabel("Log loss")
plt.legend()
plt.title("Log loss vs training iterations")
plt.show()

plt.plot(range(epochs), train_accs, label = "Training accuracy")
plt.plot(range(epochs), test_accs, label = "Testing accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Accuracy vs training iterations")
plt.show()


y_pred_train, y_pred_test = log_reg(x_train_norm, train=False), log_reg(x_test_norm, train=False)
train_classes, test_classes = predict_class(y_pred_train), predict_class(y_pred_test)
show_confusion_matrix(y_train, train_classes, 'Training')

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