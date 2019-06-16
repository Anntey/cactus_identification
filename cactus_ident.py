
import cv2
import pandas as pd
import numpy as np
from keras import layers
from keras import models
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

# loading data
base_path = "/home/antti/Downloads/cactus/"
train_path = "/home/antti/Downloads/cactus/train/train/"
test_path = "/home/antti/Downloads/cactus/test/test/"
save_path = "/home/antti/Downloads/cactus/cactus_model.hdf5"

train_df = pd.read_csv(base_path + "train.csv")
train_df["has_cactus"] = train_df["has_cactus"].astype(str) # keras requires str

test_df = pd.read_csv(base_path + "sample_submission.csv")
test_imgs = []
test_img_ids = test_df['id'].values

for img_id in test_img_ids:
    test_imgs.append(cv2.imread(test_path + img_id))
    
test_imgs = np.asarray(test_imgs)
test_imgs = test_imgs / 255.0

# generators and data augmentation
datagen = ImageDataGenerator(
        rescale = 1./255,
        horizontal_flip = True,
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

# specifying model
model = models.Sequential()
model.add(layers.Conv2D(128, (3, 3), activation = "relu", input_shape=(32, 32, 3)))
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
        optimizer = optimizers.RMSprop(lr = 1e-4),
        loss = "binary_crossentropy",
        metrics = ["accuracy"]
)

# fitting model
callbacks_list = [
        EarlyStopping(
                monitor = "val_acc",
                patience = 12,
                restore_best_weights = True
        ),
        ReduceLROnPlateau(
                monitor = "val_acc",
                factor = 0.5,
                min_lr = 1e-6,
                patience = 3,
        ),    
        ModelCheckpoint(
                save_path,
                monitor = "val_acc",
                save_best_only = True
        )
]

history = model.fit_generator(
        train_gen,
        validation_data = val_gen,
        epochs = 32,
        steps_per_epoch = 15750 / 64,
        validation_steps = 1750 / 64,
        shuffle = True,
        callbacks = callbacks_list
)

# evaluation
history_df = pd.DataFrame(history.history)
history_df[["loss", "val_loss"]].plot()
history_df[["acc", "val_acc"]].plot()

# predictions
test_df["has_cactus"] = model.predict(test_imgs)
test_df.to_csv(base_path + "cactus_subm.csv", index = False)
