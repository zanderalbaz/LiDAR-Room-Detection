# CNN & PointNet
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape, BatchNormalization


# # GNN
# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv
# from torch_geometric.datasets import Planetoid
# import torch_geometric.transforms as T

#INPUT SHAPE: (None, 5, 2, 500)
#NUM CLASSES: 12

# ======================= CNN Function =======================    
def create_3d_cnn(input_shape, num_classes): 
    model = Sequential([
            Reshape((5, 2, 500), input_shape=input_shape),  # Ensure input shape is correct
            Conv2D(40, (2, 2), padding="same", activation="relu"),
            Conv2D(20, (2, 2), padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling2D((1, 2)),  # Pool only along the 500 dimension
    ])
    assert model.output_shape == (None, 5, 1, 20)

    model.add(Conv2D(25, (2, 5), activation="relu"))
    assert model.output_shape == (None, 4, 1, 25)
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))
    model.add(MaxPooling2D((1, 2), padding="same"))

    model.add(Conv2D(50, (2, 5), activation="relu"))
    assert model.output_shape == (None, 3, 1, 50)
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))
    model.add(MaxPooling2D((1, 2), padding="same"))

    model.add(Conv2D(50, (2, 5), activation="relu"))
    assert model.output_shape == (None, 2, 1, 50)
    model.add(BatchNormalization())
    model.add(MaxPooling2D((1, 2)))

    model.add(Conv2D(100, (1, 5), activation="relu"))  
    assert model.output_shape == (None, 2, 1, 100)
    model.add(BatchNormalization())

    model.add(Flatten())
    assert model.output_shape == (None, 200)

    model.add(Dense(100, activation="relu"))
    assert model.output_shape == (None, 100)

    model.add(Dense(num_classes, activation="softmax")) # We use softmax to identify the most likely classification
    assert model.output_shape == (None, num_classes) # We have 12 total labels (3 rooms + 4 cardinal directions each)

    model.summary()

    return model

# ======================= PointNet Function =======================
def tnet(inputs, num_features):
    # T-Net mini network for PointNet
    x = layers.Conv1D(64, 1, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(1024, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(num_features * num_features, weights=[tf.zeros([256, num_features * num_features]), tf.eye(num_features).flatten()])(x)
    init = layers.Reshape((num_features, num_features))(x)

    return init

def create_pointnet_model(num_points, num_classes):
    input_points = layers.Input(shape=(num_points, 3))

    # Input Transformer
    x = tnet(input_points, 3)
    point_cloud_transformed = layers.Dot(axes=(2, 1))([input_points, x])

    # MLP
    x = layers.Conv1D(64, 1, activation='relu')(point_cloud_transformed)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(1024, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Global Features
    global_features = layers.GlobalMaxPooling1D()(x)

    # MLP on global features
    x = layers.Dense(512, activation='relu')(global_features)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Feature Transformer
    f_transform = tnet(x, 256)
    x = layers.Dot(axes=(2, 1))([global_features, f_transform])
    x = layers.Dense(256, activation='relu')(x)

    # Segmentation layer
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    # Create model
    model = models.Model(inputs=input_points, outputs=x)

    return model

# ======================= GNN Function =======================
def create_gnn(num_features, hidden_channels, num_classes):
    class GNN(torch.nn.Module):
        def __init__(self):
            super(GNN, self).__init__()
            self.conv1 = GCNConv(num_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, num_classes)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

    return GNN()


create_3d_cnn(input_shape=(5,2,500), num_classes=12)