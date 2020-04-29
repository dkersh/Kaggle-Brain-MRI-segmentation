import matplotlib.pyplot as plt
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split

# Import Neural Networks
import multiresunet
import unet_plus_plus
import unet
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint

# Useful Definitions
def dice_loss(y_true, y_pred):
    #ypred = K.greater_equal(y_pred,0.2)
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

def bce_logdice_loss(y_true, y_pred):

    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))

filenames = glob('lgg-mri-segmentation/kaggle_3m/*/*.tif')
masknames = glob('lgg-mri-segmentation/kaggle_3m/*/*mask.tif')

imagenames = [x for x in filenames if x not in masknames]

# Load Data
all_X = []
all_Y = []

for i, f in enumerate(imagenames):
    print(('\r %d / %d' % (i, len(imagenames))), end='')
    img = cv2.imread(f)
    # Only care about "post-contrast" channel
    img = img[:,:,1] / 255
    mask_name = f.replace('.tif', '_mask.tif')
    mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
    mask = mask / 255

    img = np.expand_dims(img, axis=-1)
    mask = np.expand_dims(mask, axis=-1)

    all_X += [img]
    all_Y += [mask]

print('\n')

all_X = np.array(all_X)
all_Y = np.array(all_Y)

train_X, test_X, train_Y, test_Y = train_test_split(all_X, all_Y)



print(np.shape(train_X))
print(np.shape(train_Y))

#plt.figure(figsize=(20, 10))
#plt.subplot(121)
##plt.imshow(all_X[i])
#plt.subplot(122)
##plt.imshow(all_Y[i])
#plt.colorbar()
#plt.show()

model = unet.unet()
model.compile(optimizer='adam', loss=bce_logdice_loss, metrics=[dice_coef, 'accuracy'])

checkpoint1 = ModelCheckpoint('unet_test.h5', monitor='val_dice_coef', verbose=1, save_best_only=False, mode='max', period=1)  

history = model.fit(x=train_X, y=train_Y, epochs=10, batch_size=16, verbose=1, validation_data=(test_X, test_Y), callbacks=[checkpoint1])