import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

EXPECTED_LANDMARK_LENGTH = 42  # 21 landmarks * 2 (x and y)

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Filter incomplete samples
filtered_data = []
filtered_labels = []

for sample, label in zip(data_dict['data'], data_dict['labels']):
    if len(sample) == EXPECTED_LANDMARK_LENGTH:
        filtered_data.append(sample)
        filtered_labels.append(label)

data = np.array(filtered_data)
labels = np.array(filtered_labels)

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Test
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)