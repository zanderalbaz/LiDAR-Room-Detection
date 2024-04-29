# CNN & PointNet
import tensorflow as tf
import tensorflow.keras import layers, models

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