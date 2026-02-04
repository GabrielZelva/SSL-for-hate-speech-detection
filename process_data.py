import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

data = pd.read_csv("data/labeled_data.csv")

data = data.iloc[:, [5, 6]]

model = SentenceTransformer("all-MiniLM-L6-v2")

rows = []

for row in range(len(data)):

    embedding = model.encode(data.iloc[row, 1])
    label = data.iloc[row, 0]

    rows.append(np.append(embedding, label))

processed_data = pd.DataFrame(rows)
processed_data.to_csv('data/processed_data.csv', index = False)
