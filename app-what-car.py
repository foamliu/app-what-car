import keras
from resnet_152 import resnet152_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau

img_width, img_height = 227, 227
num_channels = 3
train_data = 'mart/standford-cars-crop/train'
valid_data = 'mart/standford-cars-crop/valid'
num_classes = 196
num_train_samples = 6549
num_valid_samples = 1595
verbose = 1
batch_size = 16
num_epochs = 100000
patience = 50

# build a classifier model
model = resnet152_model(img_height, img_width, num_channels, num_classes)

# prepare data augmentation configuration
train_data_gen = ImageDataGenerator(rescale=1 / 255.,
                                    rotation_range=20.,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.2,
                                    horizontal_flip=True)
valid_data_gen = ImageDataGenerator(rescale=1. / 255)

# callbacks
tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
log_file_path = 'training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_acc', patience=patience)
reduce_lr = ReduceLROnPlateau('val_acc', factor=0.1, patience=int(patience / 4), verbose=1)
trained_models_path = 'model'
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, monitor='val_acc', verbose=1, save_best_only=True)
callbacks = [tensor_board, model_checkpoint, csv_logger, early_stop, reduce_lr]

train_generator = train_data_gen.flow_from_directory(train_data, (img_width, img_height), batch_size=batch_size,
                                                     class_mode='categorical')
valid_generator = valid_data_gen.flow_from_directory(valid_data, (img_width, img_height), batch_size=batch_size,
                                                     class_mode='categorical')

# fine tune the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=num_train_samples // batch_size,
    validation_data=valid_generator,
    validation_steps=num_train_samples // batch_size,
    epochs=num_epochs,
    callbacks=callbacks,
    verbose=verbose)

