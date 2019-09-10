#############
# Libraries #
#############

import cv2
import pandas as pd
import numpy as np
from keras import layers
from keras import models
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

################
# Loading data #
################

base_path = "./cactus/"
train_path = ."/cactus/train/"
test_path = "./cactus/test/"

train_df = pd.read_csv(base_path + "train.csv")
train_df["has_cactus"] = train_df["has_cactus"].astype(str) # keras requires str

test_df = pd.read_csv(base_path + "sample_submission.csv")
test_imgs = []
test_img_ids = test_df['id'].values

for img_id in test_img_ids:
    test_imgs.append(cv2.imread(test_path + img_id))
    
test_imgs = np.asarray(test_imgs)
test_imgs = test_imgs / 255.0

####################################
# Generators and data augmentation #
####################################

datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range = 90,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        shear_range = 0.2,
        horizontal_flip = True,
        vertical_flip = True,
        zoom_range = 0.2,
        fill_mode = "nearest",
        validation_split = 0.1
)

train_gen = datagen.flow_from_dataframe(
        train_df,
        train_path,
        x_col = "id",
        y_col = "has_cactus",
        target_size = (32, 32),
        batch_size = 64,
        shuffle = True,        
        class_mode = "binary",
        subset = "training"
)

val_gen = datagen.flow_from_dataframe(
        train_df,
        train_path,
        x_col = "id",
        y_col = "has_cactus",
        target_size = (32, 32),
        batch_size = 64,
        shuffle = True,        
        class_mode = "binary",
        subset = "validation"
)

#################
# Specify model #
#################

model = models.Sequential()
model.add(layers.Conv2D(128, (3, 3), activation = "relu", input_shape = (32, 32, 3)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(64, (3, 3), activation = "relu"))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation = "relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.4))
model.add(layers.Dense(256, activation = "relu"))
model.add(layers.Dense(1, activation = "sigmoid"))

model.compile(
        optimizer = optimizers.Adam(lr = 1e-4, decay = 1e-6),
        loss = "binary_crossentropy",
        metrics = ["accuracy"]
)

#############
# Fit model #
#############

callbacks_list = [
        EarlyStopping(
                monitor = "val_acc",
                patience = 8,
                restore_best_weights = False
        ),
        ReduceLROnPlateau(
                monitor = "val_acc",
                factor = 0.5,
                min_lr = 1e-6,
                patience = 3,
        )  
]

history = model.fit_generator(
        train_gen,
        validation_data = val_gen,
        epochs = 32,
        steps_per_epoch = 15750 / 64,
        validation_steps = 1750 / 64,
        verbose = 2,
        shuffle = True,
        callbacks = callbacks_list
)

##############
# Evaluation #
##############

history_df = pd.DataFrame(history.history)
history_df[["loss", "val_loss"]].plot()
history_df[["acc", "val_acc"]].plot()

##############
# Prediction #
##############

test_df["has_cactus"] = model.predict(test_imgs)
test_df.to_csv(base_path + "cactus_subm.csv", index = False)
