# CNN & PointNet
import tensorflow as tf
import tensorflow.keras import layers, models

# GNN
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

# ======================= CNN Function =======================
def create_3d_cnn(input_shape, num_classes):
    model - models.Sequential()

    model.add(layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

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