import tensorflow as tf

model = tf.keras.models.load_model('data/ecgdarshan_model.h5')
model.summary()