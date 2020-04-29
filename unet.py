from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, add, BatchNormalization, UpSampling2D
from tensorflow.keras.models import Model

def unet():
    input_layer = Input((256,256,1))
    
    k = [4, 8, 16, 32, 64]
    
    # Encoder
    c1 = Conv2D(k[0], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(input_layer)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)
    
    c2 = Conv2D(k[1], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)
    
    c3 = Conv2D(k[2], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)
    
    c4 = Conv2D(k[3], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    
    c5 = Conv2D(k[4], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Conv2D(k[3], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(u6)
    u7 = concatenate([u7, c3])
    u7 = Conv2D(k[2], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u7)
    u8 = concatenate([u8, c2])
    u8 = Conv2D(k[2], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(u8)
    u9 = concatenate([u9, c1])
    u9 = Conv2D(k[0], (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)

    output = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(u9)
    
    model = Model(inputs=[input_layer],outputs=[output])
    
    return model