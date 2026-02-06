
import torch
import torch.nn as nn


class model_head:
    def __init__(self):
        self.model = nn.Sequential(nn.Linear(384, 600), nn.ReLU(), nn.Linear(600, 3))

    def train(
        self,
        train_X,
        train_Y,
        dev_X,
        dev_Y,
        max_epochs=100,
        batch_size=10,
        patience=5,
        learning_rate=1e-3,
    ):
        # Save the data
        self.train_X = train_X
        self.train_Y = train_Y
        self.dev_X = dev_X
        self.dev_Y = dev_Y

        # Config
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        batches_per_epoch = len(self.train_X) // batch_size
        loss_fn = nn.CrossEntropyLoss()
        best_dev_loss = float("inf")
        epochs_without_improvement = 0

        # Core loop
        for epoch in range(max_epochs):
            self.model.train()
            for i in range(batches_per_epoch):
                # get a batch
                start = i * batch_size
                X_batch = self.train_X[start : start + batch_size]
                Y_batch = self.train_Y[start : start + batch_size]

                # forward pass
                Y_pred = self.model(X_batch)
                loss = loss_fn(Y_pred, Y_batch)

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.model.eval()

            # Early stopping check
            with torch.no_grad():
                dev_pred = self.model(self.dev_X)
                dev_loss = loss_fn(dev_pred, self.dev_Y).item()

            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break

    def predict(
        self, test_predictors, return_probabilities=False, return_predictions=False
    ):
        self.model.eval()

        with torch.no_grad():
            logits = self.model(test_predictors)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

        if return_probabilities and return_predictions:
            return probabilities, predictions

        if return_probabilities:
            return probabilities

        if return_predictions:
            return predictions

