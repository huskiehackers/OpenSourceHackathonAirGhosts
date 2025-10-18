

import tensorflow as tf


divs = ['Hori','Vert','UArr','DArr']


model = tf.keras.models.Sequential([

        # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(50, 50, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
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
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
              
history = model.fit(train_dataset,epochs=15)

test_loss, test_acc = model.evaluate(test_dataset,verbose=2)

print('\nTest accuracy:', test_acc)

model.save("gdraw.keras")



