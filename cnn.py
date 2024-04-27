import tensorflow as tf
import tensorflow.keras import layers, models

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

# Example usage:
#         Suppose the input 3D data has the shape (64, 64, 64, 1) 
#         (like a voxel grid) and you are classifying into 10 classes

# input_shape = (64, 64, 64, 1)  ---> (depth, height, width, channels)
# num_classes = 10

# model = create_3d_cnn(input_shape, num_classes)
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# print(model.summary())