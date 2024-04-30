import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def normalize_data(data):
    max_val = np.max(data)
    min_val = np.min(data)
    normalized_data = 2 * ((data - min_val) / (max_val - min_val)) - 1
    return normalized_data

def processSet(pb_set):
    pb_set = pb_set.replace('[(', '')
    pb_set = pb_set.replace(')]', '')
    return pb_set.split('), (')

def processPoints(p_list):
    points = []
    for ps in p_list:
        points.append(processPoints_(ps))
    return np.asarray(points)

def processPoints_(ps):
    xs = []
    ys = []
    for point in ps:
        p = point.split(', ')
        x = p[0]
        xs.append(int(float(x))) #convert to float then int to catch points that were not quantized
        y = p[1]
        ys.append(int(float(y))) 
        
    points = []
    points.append(np.array(xs))
    points.append(np.array(ys))
    return np.asarray(points)

def load_and_preprocess_data(filename):
    points = pd.read_csv(filename, sep='\t', header=None, names=["point_batch1", "point_batch2", "point_batch3", "point_batch4", "point_batch5", "Room+Direction"])

    labeled_data = []
    for i in range(len(points)): 
        pb_set1 = points['point_batch1'][i]
        pb_set2 = points['point_batch2'][i]
        pb_set3 = points['point_batch3'][i]
        pb_set4 = points['point_batch4'][i]
        pb_set5 = points['point_batch5'][i]

        points1 = processSet(pb_set1)
        points2 = processSet(pb_set2)
        points3 = processSet(pb_set3)
        points4 = processSet(pb_set4)
        points5 = processSet(pb_set5)

        points_list = [points1, points2, points3, points4, points5]
        point_batches = processPoints(points_list)
        label = points["Room+Direction"][i]

        labeled_data.append((point_batches, label))

    labels = [label for _, label in labeled_data]
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    num_classes = len(encoder.classes_)
    encoded_labels = to_categorical(encoded_labels, num_classes=num_classes)

    labeled_data_encoded = []
    for i in range(len(labeled_data)):
        point_batches, _ = labeled_data[i]
        encoded_label = encoded_labels[i]
        labeled_data_encoded.append((point_batches, encoded_label))
    train_data, test_data = train_test_split(labeled_data_encoded, test_size=0.1, random_state=17, shuffle=False)
    encoder_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    print(encoder_mapping)
    return train_data, test_data, num_classes, encoder.classes_

def create_visualization(model, batch, true_class):
    predicted_probs = model.predict(batch.reshape(1, 500, 2))
    predicted_label_index = np.argmax(predicted_probs)
    predicted_class = class_names[predicted_label_index]
    xs, ys = batch
    plt.scatter(xs, ys, color='b', label='Points')
    plt.title(f'Predicted: {predicted_class}, True: {true_class}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

model = load_model("lidar_detection.h5")

train_data, test_data, num_classes, class_names = load_and_preprocess_data('labeled_points.csv')

for i in range(10):
    instance_id = np.random.randint(len(test_data))

    selected_batch = np.array(test_data[instance_id][0][np.random.randint(5)])
    true_class = class_names[np.argmax(test_data[instance_id][1])]

    create_visualization(model, selected_batch, true_class)
