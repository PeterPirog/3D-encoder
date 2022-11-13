import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,Flatten,Conv2D,AveragePooling2D
from tensorflow.keras import activations

batch_size=3
points=200

if __name__ == "__main__":
    x_train=np.random.randn(batch_size,points,3,1)

    print(x_train)

    inputs = Input(shape=(points,3,1))
    #x = Dense(4, activation=tf.nn.relu)(inputs)
    x=Conv2D(filters=16,kernel_size=(3,3),padding="same",
             input_shape=(points, 3, 1))(inputs)
    x=AveragePooling2D(pool_size=(2, 2))(x)

    x=Conv2D(filters=32,kernel_size=(3,3),padding="same",
             input_shape=(points, 3, 1))(x)
    x=AveragePooling2D(pool_size=(2, 1))(x)


    x = Flatten()(x)
    outputs = Dense(1,activation=activations.linear)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.summary()
    out=model(x_train)
    print(out)