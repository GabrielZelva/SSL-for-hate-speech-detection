# %% [md]
# # Evaluating the use of Semi-Supervised Learning
# ## for hate speech and offensive language
# *By Gabriel Pišvejc*


# Using exTwitter data, I try to assess the possibility of using non labeled data to improve a transformer-encoder based model for classification.

# In particular, we will try to predict whether posts are considered hate speech, offensive language or neither. This has a level of difficulty, as the the two categories of interest often overlap. The distinction is however very important as offensive language is but a cultural perception of certain words as less prestigious, while hate speech can go as far as to be a criminal offense in certain jurisdictions and is usually used in order to discriminate. There really is an important difference between saying *let's fucking do this* and *the fucking [ethnicity of your choice] did this*.

# --------------

# %% [md]
# First things first, we will need to load the [data](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset). While the original dataset comes with some additional information about the labeling process and the text appears in raw form, I have already pre-processed it outside of this notebook.

# Particularly, I only maintained the labels and the text, as these are the two variables of interest for this report and I already passed the raw text through the transformer blocks of the [all-MiniLM-L6-v2 model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) which was designed to return the sentence embeddings. While it would be interesting to fine-tune this model for the task at hand, I decided against it as I am working on a small cuda-less notebook. Therefore, I only used the model to get the embeddings for each datapoint and I will do the SSL experiments using a custom head for the model.

# For more information on the pre-preprocessing, see the `process_data.py` script.

# %%
import pandas as pd

data = pd.read_csv("data/processed_data.csv")
# %% [md]
# While we will of course do the traditional train/dev/test split, in this particular case this comes with a caveat. The three classes are not equally represented and it would be really easy to end up having a test or a dev set nearly (or even completely) lacking a certain label.

# A similar problem comes to the surface with the missing labels, as right now, the dataset is not missing any. It is not hard to mask them artifitially in order to run the experiments, however, at say 99% masking rate it would be really easy to deepen the already deep representation problems.

# %%

# We can see the label proportions in the full dataset
data.iloc[:, len(data.columns) - 1].value_counts(normalize=True)

# 0 - Hate speech
# 1 - Offensive language
# 2 - Neither

# %% [md]
# In order to prevent these issues, I have created 2 custom functions to extract and mask certain proportions of the data label-wise. That is, if I decide to do 50% masking, it will mask 50% of each label, rather then doing it blindly. The second function works in a similar fashion, however, it is used to create splits. Therefore if I want to create a 5% split, it will pull 5% of the datapoints from each category. 

# Having said that, we will create the test split on 10% of the full dataset before masking. The train and dev sizes will be defined dynamically, as we will try the algorithm for different proportions of the unlabeled data, however, the dev split will always be 10% of the train split size.

# For more information about these functions, see the `experiment_helpers.py`.

# %%
from experiment_helpers import mask_labels, extract_equal_proportion
import torch

data, test = extract_equal_proportion(data, proportion=0.1)

test_X = torch.tensor(test.values[:, :-1], dtype=torch.float32)
test_Y = torch.tensor(test.values[:, -1], dtype=torch.long)


# %% [md]
# Aside from the data, I need a model to play with, or to be more precise, the head of the model, as we have already established that the transformer blocks and embeddings will stay intact.

# Having said that, I defined the head class in the `model_head.py` script in order to keep this notebook clean. We can use it to get a baseline with no masked data whatsoever, that is, the ideal scenario. Additionally, I will be using Monte-Carlo estimation methods to get a better estimate for the evaluation metrics. 

# I have also written a custom evaluation function which returns the overall accuracy and per label recall. More information can once again be found in the the `experiment_helpers.py` script.

# %%
import numpy as np
from model_head import ModelHead
from torch.utils.data import WeightedRandomSampler, TensorDataset, DataLoader
from experiment_helpers import evaluate_model


baseline_acc = np.array([], dtype=float)
baseline_rec0 = np.array([], dtype=float)
baseline_rec1 = np.array([], dtype=float)
baseline_rec2 = np.array([], dtype=float)

for i in range(30):

    experiment_data = data.copy()

    experiment_data = experiment_data.sample(frac=1).reset_index(drop=True)

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

    predictions = model.predict(test_X, return_predictions=True)

    accuracy, recall_0, recall_1, recall_2 = evaluate_model(
        model=model, predictions=predictions, data=test_X, ground_truth=test_Y
    )

    baseline_acc = np.append(baseline_acc, accuracy)
    baseline_rec0 = np.append(baseline_rec0, recall_0)
    baseline_rec1 = np.append(baseline_rec1, recall_1)
    baseline_rec2 = np.append(baseline_rec2, recall_2)

print(f"Accuracy estimation: {baseline_acc.mean():.3} +- {np.std(baseline_acc):.3}")
print(f"Recall on Hate speech estimation: {baseline_rec0.mean():.3} +- {np.std(baseline_rec0):.3}")
print(f"Recall on Offensive language estimation: {baseline_rec1.mean():.3} +- {np.std(baseline_rec1):.3}")
print(f"Recall on Neither estimation: {baseline_rec2.mean():.3} +- {np.std(baseline_rec2):.3}")

# %% [md]

# We can notice that even before dealing with missing labels, the model finds it hard to deal with the data imbalance. We can therefore oversample the minority classes in order to reduce this problem.

# %%

oversamp_acc = np.array([], dtype=float)
oversamp_rec0 = np.array([], dtype=float)
oversamp_rec1 = np.array([], dtype=float)
oversamp_rec2 = np.array([], dtype=float)

for i in range(30):

    experiment_data = data.copy()

    experiment_data = experiment_data.sample(frac=1).reset_index(drop=True)

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

    oversamp_acc = np.append(oversamp_acc, accuracy)
    oversamp_rec0 = np.append(oversamp_rec0, recall_0)
    oversamp_rec1 = np.append(oversamp_rec1, recall_1)
    oversamp_rec2 = np.append(oversamp_rec2, recall_2)

print(f"Accuracy estimation: {oversamp_acc.mean():.3} +- {np.std(oversamp_acc):.3}")
print(f"Recall on Hate speech estimation: {oversamp_rec0.mean():.3} +- {np.std(oversamp_rec0):.3}")
print(f"Recall on Offensive language estimation: {oversamp_rec1.mean():.3} +- {np.std(oversamp_rec1):.3}")
print(f"Recall on Neither estimation: {oversamp_rec2.mean():.3} +- {np.std(oversamp_rec2):.3}")

# %% [md]

# While the means and standard deviations shown above do seem to show a level of improvement, the propper way to evaluate this is a statistical test. In this particular example, we will use the Wilcoxon rank-sum test:

# %%
from scipy import stats

stat, p = stats.mannwhitneyu(baseline_rec0, oversamp_rec0, alternative="less")

# I am so used to R's wilcox.test() I cannot bring myself to calling it Mann-whitney 
print(f"Wilcoxon rank-sum test p-value: {p:.3}") 


# %% [md]

# As the test seems to indicate that the oversampling is indeed helping, I will keep using it in the SSL experiments. Speaking of those, the procedure will be as follows:

# I will mask different percentages of the data and then train 2 models in each scenario. One only using the labeled data, and one leveraging the power of the self-training algorithm. I will then analyse how the performance of the two compares across the different scenarios. 


# %%
from experiment_helpers import promotion_mechanism

# I will be calling each proportion of masked data a scenario
scenarios = [0.99, 0.9, 0.75, 0.5, 0.25, 0.10, 0.01]

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

for scenario in scenarios:

    row_data = []
    # Define the scenario data
    row_data.append(scenario)

    experiment_data = data.copy()

    experiment_data = mask_labels(experiment_data, mask_probability=scenario)

    unlabeled, labeled = extract_equal_proportion(experiment_data, proportion = 1)

    train, dev = extract_equal_proportion(labeled, proportion=0.1)

    dev_X = torch.tensor(dev.values[:, :-1], dtype=torch.float32)
    dev_Y = torch.tensor(dev.values[:, -1], dtype=torch.long)

    # Test a model without SSL

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

    row_data.append(accuracy)
    row_data.append(recall_0)
    row_data.append(recall_1)
    row_data.append(recall_2)

    # <Train a model with SSL>

    while True:

        if len(unlabeled) == 0: # If there is nothing left to promote
            break

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

        unlabeled_predictors = unlabeled.iloc[:,:-1]
        unlabeled_predictors = torch.tensor(unlabeled_predictors.values, dtype=torch.float32)

        probability_matrix = model.predict(unlabeled_predictors, return_probabilities=True)

        labeled_before = len(train)

        unlabeled, train = promotion_mechanism(unlabeled, train, probability_matrix)

        labeled_after = len(train)

        if labeled_before == labeled_after: # If no point promoted, then:
            break

    accuracy, recall_0, recall_1, recall_2 = evaluate_model(
        model=model, predictions=predictions, data=test_X, ground_truth=test_Y
        )


    row_data.append(accuracy)
    row_data.append(recall_0)
    row_data.append(recall_1)
    row_data.append(recall_2)

    # deTorch the results before saving them
    row_data = [t.item() if isinstance(t, torch.Tensor) else t for t in row_data]
    
    results.loc[len(results)] = row_data


# %% [md]

# Once the results are collected, we can plot them along each other. Note that as even a single run of the above experiment takes a while on a CPU, I will not be doing hypothesis testing this time. 

# %%
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

fig, ax = plt.subplots(nrows = 2, ncols = 2)

# Accuracy
ax[0,0].set_xlim(0.01, 0.99)
ax[0,0].set_ylim(0.75, 0.9)
ax[0,0].set_xlabel("Masked percentage")
ax[0,0].set_ylabel("Accuracy")

ax[0,0].plot(results["scenario"], results["accNoSSL"], label="Without SSL")
ax[0,0].plot(results["scenario"], results["accSSL"], label="With SSL")
ax[0,0].set_xticks([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
ax[0,0].tick_params(axis="both", labelsize=7)

# Recall 0
ax[0,1].set_xlim(0.01, 0.99)
ax[0,1].set_ylim(0.1, 0.6)
ax[0,1].set_xlabel("Masked percentage")
ax[0,1].set_ylabel("Recall on Hate speech")

ax[0,1].plot(results["scenario"], results["rec0NoSSL"], label="Without SSL")
ax[0,1].plot(results["scenario"], results["rec0SSL"], label="With SSL")
ax[0,1].set_xticks([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
ax[0,1].tick_params(axis="both", labelsize=7)

# Recall 1
ax[1,0].set_xlim(0.01, 0.99)
ax[1,0].set_ylim(0.8, 1)
ax[1,0].set_xlabel("Masked percentage")
ax[1,0].set_ylabel("Recall on Offensive language")

ax[1,0].plot(results["scenario"], results["rec1NoSSL"], label="Without SSL")
ax[1,0].plot(results["scenario"], results["rec1SSL"], label="With SSL")
ax[1,0].set_xticks([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
ax[1,0].tick_params(axis="both", labelsize=7)

# Recall 2
ax[1,1].set_xlim(0.01, 0.99)
ax[1,1].set_ylim(0.5, 1)
ax[1,1].set_xlabel("Masked percentage")
ax[1,1].set_ylabel("Recall on Neither")

ax[1,1].plot(results["scenario"], results["rec2NoSSL"], label="Without SSL")
ax[1,1].plot(results["scenario"], results["rec2SSL"], label="With SSL")
ax[1,1].set_xticks([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
ax[1,1].tick_params(axis="both", labelsize=7)

handles, labels = ax[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05))

fig.suptitle("Comparison between using only on the labeles subset vs. using SSL")

plt.style.use("seaborn-v0_8-whitegrid")

plt.tight_layout()
plt.show()


# %% [md]


# While SSL seems to have improved the recall on the minority class, it did so at a cost of overall accuracy and Offensive language recall. 
