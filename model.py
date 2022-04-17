from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Flatten, BatchNormalization,Dropout
from keras.preprocessing import image

#vgg16に入力する画像データのサイズを統一
img_resize = 224

#パス
train_dir = "/content/drive/MyDrive/Code/Fox Verification/dataset/train"
validation_dir = "/content/drive/MyDrive/Code/Fox Verification/dataset/validation"

#Generatorの設定
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(img_resize, img_resize),
                                                    batch_size=32,
                                                    class_mode="categorical")
  
test_generator = test_datagen.flow_from_directory(validation_dir,
                                                  target_size=(img_resize, img_resize),
                                                  batch_size = 32,
                                                  class_mode = "categorical")

#VGG16への入力層
input_tensor = Input(shape=(img_resize, img_resize, 3))

#VGG16の定義
vgg16 = VGG16(include_top = False,
              weights="imagenet",
              input_shape=(img_resize, img_resize, 3),
              input_tensor=input_tensor)
  
#froze weights of vgg16
for layer in vgg16.layers:
  layer.trainable = False

  
#buid FC layer
fc_inputs = vgg16.output ##outputsize of vgg16 = (4,4,512)
x = Flatten()(fc_inputs)
x = Dense(1024, activation="relu",name="relu1")(x)
x = BatchNormalization()(x)
x = Dense(512, activation="relu", name="relu2")(x) 
x = Dropout(rate=0.5)(x)
outputs = Dense(3, activation="softmax", name="softmax")(x)

model = Model(inputs=vgg16.input, outputs=outputs)

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(train_generator,
                    validation_data=test_generator,
                    epochs = 100,
                    batch_size=32)

model.save("/content/drive/MyDrive/Code/Fox Verification/main")
