import tensorflow.keras as keras  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import os

# Hyperparameters
lr = 1e-2
epochs = 20
batch_size = 128
img_dims = (96, 96, 3)

# Data directories
train_dir = r'archive\Dataset\Train'
test_dir = r'archive\Dataset\Test'

# Augmenting dataset with transformations
aug = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1, height_shift_range=0.1,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest"
)

# Model definition
def build(width, height, depth, classes):
    inputShape = (height, width, depth)
    chanDim = -1

    if keras.backend.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1

    model = keras.Sequential()

    # Layer 1
    model.add(keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.BatchNormalization(axis=chanDim))
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
    model.add(keras.layers.Dropout(0.25))

    # Layer 2
    model.add(keras.layers.Conv2D(64, (3, 3), padding="same"))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.BatchNormalization(axis=chanDim))

    # Layer 3
    model.add(keras.layers.Conv2D(64, (3, 3), padding="same"))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.BatchNormalization(axis=chanDim))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    # Layer 4
    model.add(keras.layers.Conv2D(128, (3, 3), padding="same"))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.BatchNormalization(axis=chanDim))

    # Layer 5
    model.add(keras.layers.Conv2D(128, (3, 3), padding="same"))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.BatchNormalization(axis=chanDim))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    # Fully connected layers
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.5))

    # Output layer
    model.add(keras.layers.Dense(classes))
    model.add(keras.layers.Activation("sigmoid"))

    return model


# Compiling the model
model = build(width=img_dims[0], height=img_dims[1], depth=img_dims[2], classes=2)
opt = keras.optimizers.Adam(learning_rate=lr, weight_decay=lr / epochs)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Data preprocessing
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255, rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest"
)

test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_dims[0], img_dims[1]),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_dims[0], img_dims[1]),
    batch_size=batch_size,
    class_mode='binary'
)

# Training the model
H = model.fit(
    train_generator,
    validation_data=test_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    verbose=1
)

# Save the trained model
model.save('gender_detection.keras')

# Plotting training metrics
N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.savefig('plot.png')
