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
    #read in LiDAR data from directory
    datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
    #trin val split
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
    num_classes = 12

    model = create_cnn(input_shape, num_classes)

    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=0.00001, verbose=1)
    # early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    model.fit(
        train_generator,
        epochs=250,
        validation_data=val_generator,
        # callbacks=[reduce_lr]
    )
# THIS IS THE FUNCTION TO RUN OUR DEMO CNN. THIS IS NOT OUR FINAL CNN
def cnn():
    input_shape = (5, 2, 500, 1)
    num_classes = 12
    X_train, X_test, y_train, y_test = load_data('labeled_points.csv')
    model = create_3d_cnn(input_shape, num_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")

# =============================== PointNet Neural  ===============================

    # PointNet is awesome, like we talked about it handles raw point clouds, 
    # making it highly suitable for our data.

    # For preprocessing we'll just need to normalize the point cloud data. For
    # example, centering and scaling. We could also do random sampling or cropping 
    # to standardize the number of points.

    # INput Shape: (x,y,z,maybe intensity?)


    #WE DID NOT GET THIS MODEL TO WORK. THIS IS DEPRECATED
    # WE REMOVED SUPPORT FOR THIS MODEL. (WE NO LONGER USE THE CSV DIRECTLY)
def pointNet():
    points, labels = load_and_preprocess_data('labeled_points.csv')

    points_train, points_test, labels_train, labels_test = train_test_split(points, labels, test_size=0.2, random_state=42)
    
    num_points = 500
    num_features = 2
    num_classes = 12
    model = create_pointnet_model(num_points, num_features, num_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(points_train, labels_train, epochs=10, batch_size=32, validation_split=0.2)
    
    test_loss, test_acc = model.evaluate(points_test, labels_test)
    print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

# =============================== Graph Neural Network ===============================
# """
# """
# GNN Example

# THIS WAS A PROTOTYPE. THIS IS ALSO NOT WORKING.
def gnn():
    num_features = 3
    hidden_channels = 16
    num_classes = 10

    dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())

    model = create_gnn(dataset.num_node_features, hidden_channels, dataset.num_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data = dataset[0].to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

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
