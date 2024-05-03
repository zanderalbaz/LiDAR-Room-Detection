# THIS CNN WAS USED FOR OUR FIRST DEMONSTRATION
# THIS CNN


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Reshape, BatchNormalization, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping

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

def augment_data(data, noise_std=0.05, transform_scale=0.1):
    augmented_data = []
    for point_batches, label in data:
        augmented_point_batches = []
        for point_set in point_batches:
            augmented_point_set = []
            for _ in range(2):  # augment each point set 2 times
                point_set_reshaped = point_set.reshape((500, 2))  
                noisy_point_set = point_set_reshaped + np.random.normal(loc=0.0, scale=noise_std, size=point_set_reshaped.shape)
                transformed_point_set = point_set_reshaped * (1 + np.random.uniform(-transform_scale, transform_scale, size=point_set_reshaped.shape))
                
                augmented_point_set.append(normalize_data(noisy_point_set))  
                augmented_point_set.append(normalize_data(transformed_point_set)) 
            
            augmented_data.append((augmented_point_set, label))
    
    return augmented_data


points = pd.read_csv('labeled_points.csv', sep='\t', header=None, names=["point_batch1", "point_batch2", "point_batch3", "point_batch4", "point_batch5", "Room+Direction"])

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
    
train_data, test_data = train_test_split(labeled_data_encoded, test_size=0.4, random_state=17, shuffle=False)
def create_3d_cnn(input_shape, num_classes): 
    model = Sequential([
        Reshape((100, 10, 1), input_shape=input_shape), 
        Conv2D(32, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        MaxPooling2D(pool_size=(2, 2)), 
        Conv2D(16, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        MaxPooling2D(pool_size=(2, 1)),
        Conv2D(8, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        Flatten(),
        Dense(32, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    model.summary()
    return model


def train_3d_cnn(train_data, test_data, input_shape, num_classes):
    model = create_3d_cnn(input_shape, num_classes)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    augmented_train_data = augment_data(train_data)
    augmented_test_data = augment_data(test_data)
    train_instances = []
    test_instances = []

    for batch, label in augmented_train_data:
        for instance in batch:
            train_instances.append((instance, label))
    for batch, label in augmented_test_data:
        for instance in batch:
            test_instances.append((instance, label))
            
    train_batches = np.array([instance for instance, _ in train_instances])
    train_labels = np.array([label for _, label in train_instances])
    test_batches = np.array([instance for instance, _ in test_instances])
    test_labels = np.array([label for _, label in test_instances])

    # Added early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    model.fit(train_batches, train_labels, epochs=100, batch_size=64, 
              validation_data=(test_batches, test_labels),
              callbacks=[early_stopping])
    model.save("lidar_detection.h5")

train_3d_cnn(train_data, test_data, (500, 2), num_classes)
