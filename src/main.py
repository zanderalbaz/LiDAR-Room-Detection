from models import create_3d_cnn
from models import create_pointnet_model
from models import create_gnn
from models import create_cnn
import torch
import pandas as pd
import numpy as np
import re
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from torch_geometric.datasets import Planetoid
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
        xs.append(int(float(x)))
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
            for _ in range(2):
                point_set_reshaped = point_set.reshape((500, 2))  
                noisy_point_set = point_set_reshaped + np.random.normal(loc=0.0, scale=noise_std, size=point_set_reshaped.shape)
                transformed_point_set = point_set_reshaped * (1 + np.random.uniform(-transform_scale, transform_scale, size=point_set_reshaped.shape))
                
                augmented_point_set.append(normalize_data(noisy_point_set))  
                augmented_point_set.append(normalize_data(transformed_point_set)) 
            
            augmented_data.append((augmented_point_set, label))
    
    return augmented_data

def load_and_preprocess_data(filepath):
    points = pd.read_csv(filepath, sep='\t', header=None, names=["point_batch1", "point_batch2", "point_batch3", "point_batch4", "point_batch5", "Room+Direction"])

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
        
    train_data, test_data = train_test_split(labeled_data_encoded, test_size=0.5, random_state=17, shuffle=False)

    return train_data, test_data

def load_data(filepath):
    data = pd.read_csv(filepath)
    X = data[['X']].values.reshape(-1, *(5, 2, 500, 1)) # <----- Input Shape 
    y = data['y'].values
    return train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# =============================== Convolutional Neural Network ===============================
# """
#     For a 3D CNN, we ahve to make sure that the LiDAR data is voxelized, 
#     which just means converting the point cloud to a 3D grid.

#     For preprocessing we can use voxel grid. Each voxel in the grid can be
#     binary (occupied/unoccupied), or we can use other values like point 
#     density or average intesity. The input shape of the network just has to 
#     match the voxel grids dimensions (depth, heigh, width, channels)
# """
def mainCNN():
    datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

    train_generator = datagen.flow_from_directory(
    'point_images',
    target_size=(160, 160),
    batch_size=8,
    color_mode='grayscale',
    class_mode='categorical',
    subset='training'
    )

    val_generator = datagen.flow_from_directory(
    'point_images',
    target_size=(160, 160),
    batch_size=8,
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation'
    )

    input_shape = (160, 160, 1)  # Since we are using grayscale we will use 1 channel
    num_classes = 8

    model = create_cnn(input_shape, num_classes)

    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001, verbose=1)
    # early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    model.fit(
        train_generator,
        epochs=30,
        validation_data=val_generator,
        # callbacks=[reduce_lr, early_stopping]
    )

def cnn():
    input_shape = (5, 2, 500, 1)
    num_classes = 12
    
    # Load & Split the Data
    X_train, X_test, y_train, y_test = load_data('labeled_points.csv')
    
    # Create Model
    model = create_3d_cnn(input_shape, num_classes)
    
    # Compile Model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train Model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    
    # Evaluate Model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")

# =============================== PointNet Neural  ===============================

    # PointNet is awesome, like we talked about it handles raw point clouds, 
    # making it highly suitable for our data.

    # For preprocessing we'll just need to normalize the point cloud data. For
    # example, centering and scaling. We could also do random sampling or cropping 
    # to standardize the number of points.

    # INput Shape: (x,y,z,maybe intensity?)

def pointNet():
    # Load and preprocess data
    points, labels = load_and_preprocess_data('labeled_points.csv')
    
    # Split data into train and test sets
    points_train, points_test, labels_train, labels_test = train_test_split(points, labels, test_size=0.2, random_state=42)
    
    # Create PointNet model
    num_points = 500
    num_features = 2
    num_classes = 12
    model = create_pointnet_model(num_points, num_features, num_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(points_train, labels_train, epochs=10, batch_size=32, validation_split=0.2)
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(points_test, labels_test)
    print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

# =============================== Graph Neural Network ===============================
# """

# """
# GNN Example
def gnn():
    # Parameters
    num_features = 3
    hidden_channels = 16
    num_classes = 10

    # Load Data
    dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())

    # Initialize Model
    model = create_gnn(dataset.num_node_features, hidden_channels, dataset.num_classes)

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data = dataset[0].to(device)

    # Define Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Train the Model
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    # Evaluating the Model
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    accuracy = correct / int(data.test_mask.sum())
    print(f'Accuracy: {accuracy:.4f}')

# =============================== Main ===============================
def main():
    mainCNN()
    # cnn()
    # pointNet()
    # gnn()

if __name__ == "__main__":
    main()
