from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, add, BatchNormalization
from tensorflow.keras.models import Model

def multiResBlock(U, input_layer):
    # Assign a parameter W which calculates the number of filters in each conv layer.
    alpha = 1.167
    W = alpha * U
    print(W)

    conv1d_shortcut = Conv2D(int(W/6) + int(W/3) + int(W/2), (1,1), activation='relu')(input_layer)

    # 3x3 Filter
#    conv1 = Conv2D(int(W/6), (3,3), activation='relu',padding='same')(input_layer)
    conv1 = Conv2D(int(W/6), (3,3), activation='relu',padding='same')(input_layer)

    # Approximate 5x5 filter
    conv2 = Conv2D(int(W/3), (5,5), activation='relu',padding='same')(conv1)
    #conv2 = Conv2D(int(W/3), (3, 3), activation='relu',padding='same')(conv2)

    # Approximate 7x7 filter
    conv3 = Conv2D(int(W/2), (7, 7), activation='relu',padding='same')(conv2)
    #conv3 = Conv2D(int(W/2), (3, 3), activation='relu',padding='same')(conv3)
    #conv3 = Conv2D(int(W/2), (3, 3), activation='relu',padding='same')(conv3)

    conc_convs = concatenate([conv1,conv2,conv3])
    conc_convs = BatchNormalization()(conc_convs)
    output = add([conc_convs,conv1d_shortcut])
    output = BatchNormalization()(output)

    return output

def ResPath(filters, length, input_layer):
    # Length is expected to go 4, 3, 2, 1

    inp = input_layer

    for i in range(length):
        conv3x3 = Conv2D(filters,(3,3), activation='relu',padding='same')(inp)
        conv1x1 = Conv2D(filters,(1,1), activation='relu',padding='same')(inp)
        conv3x3_1x1_add = add([conv1x1,conv3x3])
        output = BatchNormalization()(conv3x3_1x1_add)
        inp = output

    return output

def multiResUNetModel():
    input_layer = Input((240,320,1))
    
    mres1 = multiResBlock(16,input_layer)
    p1 = MaxPooling2D((2,2))(mres1)
    mres1 = ResPath(16,4,mres1)

    mres2 = multiResBlock(32, p1)
    p2 = MaxPooling2D((2, 2))(mres2)
    mres2 = ResPath(32, 3, mres2)

    mres3 = multiResBlock(64, p2)
    p3 = MaxPooling2D((2, 2))(mres3)
    mres3 = ResPath(64, 2, mres3)

    mres4 = multiResBlock(128, p3)
    p4 = MaxPooling2D((2, 2))(mres4)
    mres4 = ResPath(128, 1, mres4)

    mres5 = multiResBlock(256, p4)

    up6 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(mres5)
    up6 = concatenate([up6,mres4])
    up6 = multiResBlock(128,up6)

    up7 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(up6)
    up7 = concatenate([up7, mres3])
    up7 = multiResBlock(64, up7)

    up8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(up7)
    up8 = concatenate([up8, mres2])
    up8 = multiResBlock(32, up8)

    up9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(up8)
    up9 = concatenate([up9, mres1])
    up9 = multiResBlock(16, up9)

    output = Conv2D(1, (1,1), activation='sigmoid')(up9)

    model = Model(inputs=[input_layer],outputs=[output])

    return model
