import tensorflow as tf


class Normalize(tf.Module):
    def __init__(self, x):
        # Initialize the mean and standard deviation for normalization
        super().__init__()
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
        super().__init__()
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


    def predict_class(self, y_pred, thresh=0.5):
        # Return a tensor with  `1` if `y_pred` > `0.5`, and `0` otherwise
        return tf.cast(y_pred > thresh, tf.float32)


    def accuracy(self, y_pred, y):
        # Return the proportion of matches between `y_pred` and `y`
        y_pred = tf.math.sigmoid(y_pred)
        y_pred_class = self.predict_class(y_pred)
        check_equal = tf.cast(y_pred_class == y,tf.float32)
        acc_val = tf.reduce_mean(check_equal)
        return acc_val

    def log_loss(self, y_pred, y):
        # Compute the log loss function
        ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_pred)
        return tf.reduce_mean(ce)

    def train(self, epochs, train_dataset, valid_dataset, learning_rate = 0.01):

        train_losses, valid_losses = [], []
        train_accs, valid_accs = [], []
        # Set up the training loop and begin training
        for epoch in range(epochs):
            batch_losses_train, batch_accs_train = [], []
            batch_losses_valid, batch_accs_valid = [], []

            # Iterate over the training data
            for x_batch, y_batch in train_dataset:
                with tf.GradientTape() as tape:
                    y_pred_batch = self(x_batch)
                    batch_loss = self.log_loss(y_pred_batch, y_batch)
                batch_acc = self.accuracy(y_pred_batch, y_batch)
                # Update the parameters with respect to the gradient calculations
                grads = tape.gradient(batch_loss, self.variables)


                for g, v in zip(grads, self.variables):
                    v.assign_sub(learning_rate * g)
                # Keep track of batch-level training performance
                batch_losses_train.append(batch_loss)
                batch_accs_train.append(batch_acc)

            # Iterate over the testing data
            for x_batch, y_batch in valid_dataset:
                y_pred_batch = self(x_batch)
                batch_loss = self.log_loss(y_pred_batch, y_batch)
                batch_acc = self.accuracy(y_pred_batch, y_batch)
                # Keep track of batch-level testing performance
                batch_losses_valid.append(batch_loss)
                batch_accs_valid.append(batch_acc)

            # Keep track of epoch-level model performance
            train_loss, train_acc = tf.reduce_mean(batch_losses_train), tf.reduce_mean(batch_accs_train)
            valid_loss, valid_acc = tf.reduce_mean(batch_losses_valid), tf.reduce_mean(batch_accs_valid)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)
            if epoch % 20 == 0:
                print(f"Epoch: {epoch}, Training log loss: {train_loss:.3f}")

        return train_losses, valid_losses, train_accs, valid_accs
