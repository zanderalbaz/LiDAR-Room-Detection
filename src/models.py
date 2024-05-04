# CNN & PointNet
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
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

# ======================= NEW CNN Function =======================
# We can possibly use these to improve training dynamically:

# from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001, verbose=1)
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# The callback 'ReduceLROnPlateau' dynamically adjusts the learning rate. We can also use 'EarlyStopping'
# to prevent any overfitting...

def create_cnn(input_shape, num_classes): 
    model = Sequential([
        Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),

        Dense(512, activation='relu'),
        Dropout(0.5),
        BatchNormalization(),

        Dense(num_classes, activation='softmax')
    ])

    model.summary()
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# ======================= CNN Function =======================    
def create_3d_cnn(input_shape, num_classes):  #
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
# """
#     The sole purpose of the Transformation Network, is to learn an optimal 
#     spatial transformation of the input data (Whats most important.)
# """

from keras.initializers import Constant

def tnet(input_points, num_features):
    transform = Dense(64, activation='relu')(input_points)
    transform = Dense(128, activation='relu')(transform)
    transform = Dense(256, activation='relu')(transform)
    transform = Dense(512, activation='relu')(transform)
    transform = Dense(256, activation='relu')(transform)
    transform = Dense(128, activation='relu')(transform)
    
    # Ensure the output size matches num_features * num_features
    output_size = num_features * num_features
    transform = Dense(output_size, 
                      kernel_initializer='zeros', 
                      bias_initializer=Constant(np.eye(num_features).flatten()))(transform)
    transform = Reshape((num_features, num_features))(transform)  # Reshape to (num_features, num_features)
    
    return transform

def create_pointnet_model(num_points, num_features, num_classes):
    input_points = Input(shape=(num_points, num_features))
    transformed = tnet(input_points, num_features)
    x = Dense(64, activation='relu')(transformed)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = MaxPooling1D(pool_size=num_points)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
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


# create_3d_cnn(input_shape=(5,2,500), num_classes=12)