import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.vgg19 import VGG19
from keras.utils.vis_utils import plot_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
# %% 
train_dir = '../dataset/split/train'
val_dir = '../dataset/split/val'
test_dir = '../dataset/split/test'
# %%
diseases_name = []
for image_class in os.listdir(train_dir):
    diseases_name.append(image_class)
print(diseases_name)
print(f'Total Disease: {len(diseases_name)}')
# %%
train_data = image_dataset_from_directory(
    train_dir,
    label_mode="binary",
    image_size=(224, 224),
    batch_size=32,
    shuffle=True,
    seed=42
)
test_data = image_dataset_from_directory(
    test_dir,
    label_mode="binary",
    image_size=(224, 224),
    batch_size=32,
    shuffle=True, 
    seed=42
)
val_data = image_dataset_from_directory(
    val_dir,
    label_mode="binary",
    image_size=(224, 224),
    batch_size=32,
    shuffle=True, seed=42
)
# %%
model = Sequential()

pretrained_model = VGG19(
    include_top=False,
    input_shape=(224, 224, 3),
    pooling='max',
    weights='imagenet'
)

pretrained_model.trainable = False


pretrained_model = Model(
    inputs=pretrained_model.inputs,
    outputs=pretrained_model.layers[-2].output
)
model.add(pretrained_model)

model.add(BatchNormalization())

model.add(GlobalAveragePooling2D())

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()
# %%
plot_model(model,
            to_file='../model_arc/VGG19.png',
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=False,
            )
# %%
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                              min_delta=0.0001)
# %%
model.compile(loss=tf.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy',])
# %%
hist = model.fit(train_data, epochs=50, validation_data=val_data,
                 callbacks=[early_stop])
test_loss, test_acc = model.evaluate(test_data)
# %%
with open("../graph/VGG19.txt", "w") as outfile:
    outfile.write(f"vgg19_train_accuracy={hist.history['accuracy']}\nvgg19_validation_accuracy={hist.history['val_accuracy']}\nvgg19_train_loss={hist.history['loss']}\nvgg19_validation_loss={hist.history['val_loss']}\nvgg19_test_loss={test_loss}\nvgg19_test_accuracy={test_acc}\n")
# %%
model.save(f'../saved_models/VGG19.h5')
# %%
print(f'Test Loss: {test_loss} Test Accuracy: {test_acc}')
# %%
test_images = []
test_labels = []
for images, labels in test_data:
    test_images.append(images.numpy())
    test_labels.append(labels.numpy())
test_images = np.concatenate(test_images)
test_labels = np.concatenate(test_labels)

# Get model predictions
predictions = model.predict(test_images)
predicted_labels = np.round(predictions).flatten()

# Calculate confusion matrix"
cm = confusion_matrix(test_labels, predicted_labels)
print("Confusion Matrix:")
print(cm)

# Calculate precision
precision = precision_score(test_labels, predicted_labels)
print("Precision:", precision)

# Calculate recall
recall = recall_score(test_labels, predicted_labels)
print("Recall:", recall)