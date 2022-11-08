def predict(path):
    import tensorflow as tf
    from numpy import array, round
    model = tf.keras.models.load_model('models\model-1.h5')
    image = tf.keras.utils.load_img(path, target_size=(150, 150))
    image = array(image)
    image = tf.expand_dims(image, axis=0)
    probs = model.predict(image)
    pred = int(round(probs))
    return pred
