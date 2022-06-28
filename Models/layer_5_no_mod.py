from tensorflow.keras import models, layers

def create_model():

  inp = layers.Input(shape=(256, 256, 1))

  conv1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inp)
  conv1 = layers.Conv2D(16, 3, activation='relu', padding='same')(conv1)
  pool1 = layers.MaxPool2D(2)(conv1)

  conv2 = layers.Conv2D(32, 3, activation='relu', padding='same')(pool1)
  conv2 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv2)
  pool2 = layers.MaxPool2D(2)(conv2)

  conv3 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool2)
  conv3 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv3)
  pool3 = layers.MaxPool2D(2)(conv3)

  conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool3)
  conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
  pool4 = layers.MaxPool2D(2)(conv8)

  conv10 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool4)
  conv10 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv10)
  pool5 = layers.MaxPool2D(2)(conv10)

  conv4_ = layers.Conv2D(450, 3, activation='relu', padding='same')(pool5)
  conv4 = layers.Conv2D(450, 3, activation='relu', padding='same')(conv4_)

  dconv5 = layers.Conv2DTranspose(256, 3, strides=2, activation='relu', padding='same')(conv4)
  conc5 = layers.concatenate([dconv5, conv10])
  conv11 = layers.Conv2D(256, 3, activation='relu', padding='same')(conc5)
  conv11 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv11)

  dconv4 = layers.Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same')(conv11)
  conc4 = layers.concatenate([dconv4, conv8])
  conv9 = layers.Conv2D(128, 3, activation='relu', padding='same')(conc4)
  conv9 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv9)

  dconv3 = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(conv9)
  conc3 = layers.concatenate([dconv3, conv3])
  conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(conc3)
  conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv5)

  dconv2 = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(conv5)
  conc2 = layers.concatenate([dconv2, conv2])
  conv6 = layers.Conv2D(32, 3, activation='relu', padding='same')(conc2)
  conv6 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv6)

  dconv1 = layers.Conv2DTranspose(16, 3, strides=2, activation='relu', padding='same')(conv6)
  conc1 = layers.concatenate([dconv1, conv1])
  conv7 = layers.Conv2D(16, 3, activation='relu', padding='same')(conc1)
  conv7 = layers.Conv2D(16, 3, activation='relu', padding='same')(conv7)

  outp1 = layers.Conv2D(8, 1, name='output1', activation='sigmoid', padding='same')(conv7)

  flat = layers.Flatten()(conv4_)
  dense1 = layers.Dense(100, activation='relu')(flat)
  dense1_ = layers.Dense(80, activation='relu')(dense1)
  dense1 = layers.Dense(60, activation='relu')(dense1_)

  dense2 = layers.Dense(40, activation='relu')(dense1)
  dense2 = layers.Dense(20, activation='relu')(dense2)
  outp2 = layers.Dense(3, name='output2', activation='sigmoid')(dense2)

  dense3 = layers.Dense(40, activation='relu')(dense1)
  dense3 = layers.Dense(20, activation='relu')(dense3)
  dense3 = layers.Dense(10, activation='relu')(dense3)
  outp3 = layers.Dense(1, name='output3', activation='relu')(dense3)

  model = models.Model(inputs=inp, outputs=[outp1, outp2, outp3])
  
  return model


