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


# %% [md]
# Aside from the data, I need a model to play with, or to be more precise, the head of the model, as we have already established that the transformer blocks and embeddings will stay intact.

# Having said that, I defined the head class in the `model_head.py` script in order to keep this notebook clean. We can use it to get a baseline with no masked data whatsoever, that is, the ideal scenario.

# %%
from model_head import ModelHead
from torch.utils.data import WeightedRandomSampler, TensorDataset, DataLoader

experiment_data = data.copy()

train, dev = extract_equal_proportion(experiment_data, proportion=0.1)
dev_X = torch.tensor(dev.values[:, :-1], dtype=torch.float32)
dev_Y = torch.tensor(dev.values[:, -1], dtype=torch.long)

train_X = train.iloc[:, :-1]
train_Y = train.iloc[:, -1]

train_X = torch.tensor(train_X.values, dtype=torch.float32)
train_Y = torch.tensor(train_Y.values, dtype=torch.long)

train_dataset = TensorDataset(train_X, train_Y)
train_loader = DataLoader(train_dataset, batch_size=32)

model = ModelHead()

model.train(train_loader, dev_X, dev_Y)

# %% [md]
# I have also written a custom evaluation function which returns the overall accuracy and per label recall. More information can once again be found in the the `experiment_helpers.py` script.

# %%
from experiment_helpers import evaluate_model

predictions = model.predict(test_X, return_predictions=True)

accuracy, recall_0, recall_1, recall_2 = evaluate_model(
    model=model, predictions=predictions, data=test_X, ground_truth=test_Y
)

print(f"Overall accuracy: {accuracy:.3f}")
print(f"Recall on hate speech: {recall_0:.3f}")
print(f"Recall on offensive language: {recall_1:.3f}")
print(f"Recall on neither: {recall_2:.3f}")

# %% [md]
# We can notice that even before dealing with missing labels, the model finds it hard to deal with the data imbalance. We can therefore oversample the minority classes in order to reduce this problem. 

# %%
experiment_data = data.copy()

train, dev = extract_equal_proportion(experiment_data, proportion=0.1)

dev_X = torch.tensor(dev.values[:, :-1], dtype=torch.float32)
dev_Y = torch.tensor(dev.values[:, -1], dtype=torch.long)

train_X = train.iloc[:, :-1]
train_Y = train.iloc[:, -1]

train_X = torch.tensor(train_X.values, dtype=torch.float32)
train_Y = torch.tensor(train_Y.values, dtype=torch.long)

class_counts = torch.bincount(train_Y)
class_weights = 1.0 / class_counts.float()
sample_weights = class_weights[train_Y]

sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(sample_weights), replacement=True
)

train_dataset = TensorDataset(train_X, train_Y)
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)

model = ModelHead()

model.train(train_loader, dev_X, dev_Y)

predictions = model.predict(test_X, return_predictions=True)

accuracy, recall_0, recall_1, recall_2 = evaluate_model(
    model=model, predictions=predictions, data=test_X, ground_truth=test_Y
)

print(f"Overall accuracy: {accuracy:.3f}")
print(f"Recall on hate speech: {recall_0:.3f}")
print(f"Recall on offensive language: {recall_1:.3f}")
print(f"Recall on neither: {recall_2:.3f}")

# %% [md]

# Aaaand, I acomplished absolutely nothing with this. I assume I could probably fix this if I torture the loss function enough, adjust the decision treshold or do something to the embeddings but... This was meant to be a 15 minute exercise where I would use a library to do everything for me and I have spent... way too much time... on this already. ~40% recall on a 5% minority class is fine, so I am just going to continue. Also, I will keep oversampling for the SSL, because I would like to avoid the minoriry class being an even bigger problem when I start masking. 

# %%
# I will be calling each proportion of masked data a scenario
scenarios = [0.9, 0.75, 0.5, 0.25, 0.10]

results = pd.DataFrame(
    columns=[
        "scenario",
        "accNoSSL",
        "rec0NoSSL",
        "rec1NoSSL",
        "rec2NoSSL",
        "accSSL",
        "rec0SSL",
        "rec1SSL",
        "rec2SSL",
    ]
)

index = 0
for scenario in scenarios:

    # Define the scenario data
    results.iloc[index, 0] = scenario

    experiment_data = data.copy()

    experiment_data = mask_labels(experiment_data, mask_probability=scenario)

    train, dev = extract_equal_proportion(experiment_data, proportion=0.1)

    dev_X = torch.tensor(dev.values[:, :-1], dtype=torch.float32)
    dev_Y = torch.tensor(dev.values[:, -1], dtype=torch.long)

    train_X = train.iloc[:, :-1]
    train_Y = train.iloc[:, -1]

    train_X = torch.tensor(train_X.values, dtype=torch.float32)
    train_Y = torch.tensor(train_Y.values, dtype=torch.long)

    class_counts = torch.bincount(train_Y)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[train_Y]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_dataset = TensorDataset(train_X, train_Y)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)

    model = ModelHead()

    model.train(train_loader, dev_X, dev_Y)

    predictions = model.predict(test_X, return_predictions=True)

    accuracy, recall_0, recall_1, recall_2 = evaluate_model(
        model=model, predictions=predictions, data=test_X, ground_truth=test_Y
    )

    results.iloc[index, 1] = accuracy
    results.iloc[index, 2] = recall_0
    results.iloc[index, 3] = recall_1
    results.iloc[index, 4] = recall_2

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

    index += 1


# %%
