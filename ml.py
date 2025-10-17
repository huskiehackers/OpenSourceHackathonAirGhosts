

import tensorflow as tf


divs = ['Hori','Vert','UArr','DArr']


model = tf.keras.models.Sequential([

    tf.keras.layers.Flatten(input_shape=(50, 50)),
    tf.keras.layers.Dense(128, activation='softmax'),
    tf.keras.layers.Dense(4)
    ])
    

train_dataset = tf.keras.utils.image_dataset_from_directory(
    "train",
    image_size=(50, 50),   # resize to same shape as Fashion MNIST
    color_mode="grayscale",  # or "rgb"
    batch_size=32
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    "test",
    image_size=(50, 50),
    color_mode="grayscale",
    batch_size=32
)

# Optional: normalize images (scale pixel values 0–1)
train_dataset = train_dataset.map(lambda x, y: (x / 255.0, y))
test_dataset = test_dataset.map(lambda x, y: (x / 255.0, y))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
              
model.fit(train_dataset,epochs=10)

test_loss, test_acc = model.evaluate(train_dataset,verbose=2)

print('\nTest accuracy:', test_acc)

model.save("gdraw.keras")

