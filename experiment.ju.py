# %% [md]
# # Evaluating the use of Semi-Supervised Learning
# ## for hate speech and offensive language
# *By Gabriel Pišvejc*


# Using exTwitter data, I try to assess the possibility of using non labeled data to improve a transformer-encoder based model for classification.

# In particular, we will try to predict whether posts are considered hate speech, offensive language or neither. This has a level of difficulty, as the the two categories of interest of interest often overlap. The distinction is however very important as offensive language is but a cultural perception of certain words as less prestigious, while hate speech can go as far as to be a criminal offense in certain jurisdictions and is usually used in order to discriminate. There is an important difference between saying *let's fucking do this* and *the fucking [ethnicity of your choice] did this*.

# --------------

# %% [md]
# First things first, we will need to load the [data](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset). While the original dataset comes with some additional information about the labeling process and the text appears in raw form, I have already preprocessed it outside of this notebook.

# In particular, I only maintained the labels and the text, as these are the two variables of interest for this particular report. Aditionally, I already passed the raw text through the transformer blocks of the [all-MiniLM-L6-v2 model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) which was designed to return the sentence embeddings. While it would be interesting to fine-tune this model for the task at hand, I decided against it as I am working on a small cuda-less notebook. Therefore, I only used the model to get the embeddings for each datapoint and we will do the SSL using a custom head for the model.

# For more information on the data preprocessing, see the `process_data.py` script.

# %%
import pandas as pd

data = pd.read_csv("data/processed_data.csv")
# %% [md]
# While we will of course do the traditional train/dev/test split, in this particular example this comes with a caveat. The three classes are not equally represented and it would be really easy to end up having a test or a dev set nearly (or even completely) lacking a certain label.

# A similar problem comes to the surface with the missing labels, as right now, the dataset is not missing any. It is not hard to mask them artifitially in order to run the experiments, however, at say ~90% masking rate it would be really easy to deepen the already deep representation problems.

# %%

# We can see the label proportions in the full dataset
data.iloc[:, len(data.columns) - 1].value_counts(normalize=True)

# 0 - Hate speech
# 1 - Offensive language
# 2 - Neither

# %% [md]
# In order to prevent these issues, I have created 2 custom functions to extract and mask certain proportions of the data label-wise. That is, if I decide to do 50% masking, it will mask 50% of each label, rather then doing it blindly. The same applies to creating splits. Therefore, there will be no need to worry about label representation in any split.

# Having said that, we will create the test split on 10% of the full dataset before masking. The train and dev sizes will be defined dynamically, as we will try the algorythm for different proportions of unlabeled data, however, the dev split will always be 10% of the train split size.

# For more information about the functions, see the `experiment_helpers.py`.

# %%
from experiment_helpers import mask_labels, extract_equal_proportion
import torch

data, test = extract_equal_proportion(data, proportion=0.1)

test_X = torch.tensor(test.values[:, :-1], dtype=torch.float32)
test_Y = torch.tensor(test.values[:, -1], dtype=torch.long)


# %%
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


# %%
# Testing if the model works

experiment_data = data.copy()

train, dev = extract_equal_proportion(experiment_data, proportion=0.1)

train_X = torch.tensor(train.values[:, :-1], dtype=torch.float32)
train_Y = torch.tensor(train.values[:, -1], dtype=torch.long)

dev_X = torch.tensor(dev.values[:, :-1], dtype=torch.float32)
dev_Y = torch.tensor(dev.values[:, -1], dtype=torch.long)

model = model_head()

model.train(train_X, train_Y, dev_X, dev_Y)

model.predict(test_X[0].unsqueeze(0), return_probabilities=True)

# %%
# I will be calling each proportion of masked data a scenario
scenarios = [0.9, 0.75, 0.5, 0.25, 0.10]

# Create an empty dataframe with the following data
# - scenario
# - accuracy without SLL
# - recall for each label without SSL
# - accuracy with SLL
# - recall for each label with SSL

for scenario in scenarios:
    # Define the data situation for the scenario

    experiment_data = data.copy()

    experiment_data = mask_labels(experiment_data, mask_probability=scenario)
    train, dev = extract_equal_proportion(experiment_data, proportion=0.1)

    dev_X = torch.tensor(dev.values[:, :-1])
    dev_Y = torch.tensor(dev.values[:, -1])

    unlabeled_data, labeled_data = extract_equal_proportion(
        experiment_data, proportion=1
    )

    # <Train a model without SSL>

    # Save the accuracy and recall per label

    # </Train a model without SSL>

    # <Train a model with SSL>

    while True:
        unlabeled_predictors = torch.tensor(dev.unlabeled_data[:, :-1])
        labeled_predictors = torch.tensor(dev.labeled_data[:, :-1])
        labels = torch.tensor(dev.labeled_data[:, -1])

        # Train the model on labeled

        # Predict on unlabeled

        # if any <90% probs:

        # take them into a new df

        # Promote them

        # Join this df into the labeled

        # else:

        # break

    # Once we get here, this is the final model.
    # Evaluate
    # </Train a model with SSL>
    # Write a line into the results table


# %%
