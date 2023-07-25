import seaborn as sns
import sklearn.metrics as sk_metrics
from matplotlib import pyplot as plt


def corr_plot(train_data):
    f, ax = plt.subplots(figsize=(10, 8))
    corr = train_data.corr(method="kendall")
    sns.heatmap(corr,
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                vmin=-1.0, vmax=1.0,
                square=True, ax=ax)
    plt.show()

def confusion_matrix_plot(y, y_classes, typ):
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


def range_train_test_plot(epochs, train, test, what):
    plt.plot(range(epochs), train, label = f"Training {what}")
    plt.plot(range(epochs), test, label = f"Testing {what}")
    plt.xlabel("Epoch")
    plt.ylabel(f"{what}")
    plt.legend()
    # plt.title("Log loss vs training iterations")
    plt.show()