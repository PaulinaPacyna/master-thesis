#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
from models import FCN_model
from preprocessing import ConstantLengthDataGenerator
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import mlflow
from preprocessing.utils import plot
import tensorflow as tf
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()
mlflow.set_experiment("FCN")
mlflow.tensorflow.autolog()


# In[2]:


data_path = "./data"
X, y = np.load(f"{data_path}/X.npy", allow_pickle=True), np.load(f"{data_path}/y.npy")


# In[3]:


mask = np.char.startswith(y, "GunPoint").reshape(-1)
y = y[mask, :]
X = X[mask]
mlflow.log_param("y.unique", np.unique(y))


# In[4]:


y_encoder = sklearn.preprocessing.OneHotEncoder(categories="auto")
y = y_encoder.fit_transform(y.reshape(-1, 1)).toarray()
mlflow.log_param("y.shape", y.shape)


# In[5]:


number_of_classes = y.shape[1]
initial_learning_rate = 1e-4
output_directory = f"{data_path}/models/fcn/outputs"
batch_size = 1024
os.makedirs(output_directory, exist_ok=True)


# In[6]:


input_layer = keras.layers.Input(shape=(None, 1))
fcn_model = FCN_model(number_of_classes=number_of_classes, parameters=0.25)(input_layer)
model = keras.models.Model(inputs=input_layer, outputs=fcn_model)

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate, decay_steps=3, decay_rate=1
)
model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(lr_schedule),
    metrics=["accuracy"],
    run_eagerly=True,
)


# In[7]:


model.summary()


# In[8]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)


# In[9]:


kwargs = {"min_length": 256, "max_length": 256}
data_generator_train = ConstantLengthDataGenerator(
    X_train, y_train, batch_size=batch_size, padding_probability=0.5, cutting_probability=0.5, augmentation_probability=0.5, **kwargs
)
data_generator_val = ConstantLengthDataGenerator(
    X_val, y_val, batch_size=len(y_val)*5, **kwargs
)
validation_data = next(data_generator_val)


# In[16]:


history = model.fit(data_generator_train, epochs=50, validation_data=validation_data
)


# In[ ]:


figure = plt.figure()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "validation"], loc="upper left")


# In[ ]:


mlflow.log_figure(figure, "data/figures/acc.png")


# In[ ]:


figure = plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "validation"], loc="upper left")


# In[ ]:


mlflow.log_figure(figure, "data/figures/loss.png")


# In[ ]:


mlflow.log_artifact("models")
mlflow.log_artifact("preprocessing")


# In[ ]:


mlflow.end_run()

