from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, add, BatchNormalization, Dropout
from tensorflow.keras.models import Model

def conv_block(input_tensor,filter_num,filter_size):

    x = Conv2D(filter_num,filter_size,activation='relu',padding='same',kernel_initializer='he_normal')(input_tensor)
    x = Dropout(0.1)(x)
    x = Conv2D(filter_num, filter_size, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Dropout(0.1)(x)

    return x

'''
Standard UNet++
Total Parameters: lots
'''

def unetpp():

    filt_num = [32,64,128,256,512]

    input_tensor = Input(shape=(240,320,1))

    conv1_1 = conv_block(input_tensor,filt_num[0],(3,3))
    pool_1 = MaxPooling2D((2,2), strides=(2,2))(conv1_1)

    conv2_1 = conv_block(pool_1,filt_num[1],(3,3))
    pool_2 = MaxPooling2D((2,2), strides=(2,2))(conv2_1)

    up1_1 = Conv2DTranspose(filt_num[0],(2,2),strides=(2,2),padding='same')(conv2_1)
    conv1_2 = concatenate([up1_1,conv1_1])
    conv1_2 = conv_block(conv1_2,filt_num[0],(3,3))

    conv3_1 = conv_block(pool_2,filt_num[2],(3,3))
    pool_3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3_1)

    up2_1 = Conv2DTranspose(filt_num[1],(2,2),strides=(2,2),padding='same')(conv3_1)
    conv2_2 = concatenate([up2_1,conv2_1])
    conv2_2 = conv_block(conv2_2,filt_num[0],(3,3))

    up1_2 = Conv2DTranspose(filt_num[0],(2,2),strides=(2,2),padding='same')(conv2_2)
    conv1_3 = concatenate([up1_2,conv1_1,conv1_2])

    conv4_1 = conv_block(pool_3,filt_num[3],(3,3))
    pool_4 = MaxPooling2D((2,2), strides=(2,2))(conv4_1)

    up3_1 = Conv2DTranspose(filt_num[2],(2,2),strides=(2,2),padding='same')(conv4_1)
    conv3_2 = concatenate([up3_1,conv3_1])
    conv3_2 = conv_block(conv3_2, filt_num[2], (3, 3))

    up2_2 = Conv2DTranspose(filt_num[1],(2,2),strides=(2,2),padding='same')(conv3_2)
    conv2_3 = concatenate([up2_2,conv2_2,conv2_1])
    conv2_3 = conv_block(conv2_3, filt_num[1], (3, 3))

    up1_3 = Conv2DTranspose(filt_num[0],(2,2),strides=(2,2),padding='same')(conv2_3)
    conv1_4 = concatenate([up1_3,conv1_3,conv1_2,conv1_1])
    conv1_4 = conv_block(conv1_4, filt_num[0], (3, 3))

    conv5_1 = conv_block(pool_4,filt_num[4],(3,3))

    up4_1 = Conv2DTranspose(filt_num[3],(2,2),strides=(2,2),padding='same')(conv5_1)
    conv4_2 = concatenate([up4_1,conv4_1])
    conv4_2 = conv_block(conv4_2,filt_num[3],(3,3))

    up3_2 = Conv2DTranspose(filt_num[2],(2,2),strides=(2,2),padding='same')(conv4_2)
    conv3_3 = concatenate([up3_2,conv3_2,conv3_1])
    conv3_3 = conv_block(conv3_3, filt_num[2], (3, 3))

    up2_3 = Conv2DTranspose(filt_num[1],(2,2),strides=(2,2),padding='same')(conv3_3)
    conv2_4 = concatenate([up2_3,conv2_3,conv2_2,conv2_1])
    conv2_4 = conv_block(conv2_4, filt_num[1], (3, 3))

    up1_4 = Conv2DTranspose(filt_num[0],(2,2),strides=(2,2),padding='same')(conv2_4)
    conv1_5 = concatenate([up1_4,conv1_4,conv1_3,conv1_2,conv1_1])
    conv1_5 = conv_block(conv1_5, filt_num[0], (3, 3))

    output = Conv2D(1,(1,1), activation='sigmoid')(conv1_5)

    model = Model(inputs=input_tensor,outputs=output)

    return model
