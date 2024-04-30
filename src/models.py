# CNN & PointNet
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# GNN
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

#INPUT SHAPE: (None, 5, 2, 500)
#NUM CLASSES: 12

# ======================= CNN Function =======================
def create_cnn(input_shape, num_classes): 
    model = Sequential([
            Reshape((5, 2, 500, 1), input_shape=(5, 2, 500)),  # Reshape input to match the desired input size (5, 2, 500)
            Conv2D(10, (2, 5), activation="relu"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
    ])
    assert model.output_shape == (None, 2, 250, 10)

    model.add(Conv2D(25, (2, 5), activation="relu"))
    assert model.output_shape == (None, 1, 125, 25)
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))
    model.add(MaxPooling2D((2, 2), padding="same"))

    model.add(Conv2D(50, (2, 5), activation="relu"))
    assert model.output_shape == (None, 1, 63, 50)
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.5))
    model.add(MaxPooling2D((2, 2), padding="same"))

    model.add(Conv2D(50, (2, 5), activation="relu"))
    assert model.output_shape == (None, 1, 32, 50)
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(100, (1, 5), activation="relu"))  
    assert model.output_shape == (None, 1, 16, 100)
    model.add(BatchNormalization())

    model.add(Flatten())
    assert model.output_shape == (None, 1600) 

    model.add(Dense(100, activation="relu"))
    assert model.output_shape == (None, 100)

    model.add(Dense(12, activation="softmax")) # We use softmax to identify the most likely classification
    assert model.output_shape == (None, 12) # We have 12 total labels

    model.summary()

    return model

# ======================= PointNet Function =======================
# """
#     The sole purpose of the Transformation Network, is to learn an optimal 
#     spatial transformation of the input data (Whats most important.)
# """

def tnet(inputs, num_features):
    x = Dense(64, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    transform = Dense(num_features * num_features, kernel_initializer='zeros', bias_initializer=tf.keras.initializers.Constant(np.eye(num_features).flatten()))(x)
    transform = Reshape((num_features, num_features))(transform)
    transformed = Multiply()([inputs, transform])
    return transformed

def create_pointnet_model(num_points, num_classes):
    input_points = Input(shape=(num_points, num_features))
    transformed = tnet(input_points, num_features)
    x = Dense(64, activation='relu')(transformed)
    x = Dense(128, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = MaxPool1D(pool_size=num_points)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_points, outputs=output)
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