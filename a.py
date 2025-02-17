import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import torch


def main():
    st.title("My First Streamlit App")
    st.write("Upload an image for classification!")


    file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if file:
        image= Image.open(file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        resized_image = image.resize((32, 32))
        img_array = np.array(resized_image)/255.0
        img_array = img_array.reshape(1, 32, 32, 3)
        # img_array = np.expand_dims(img_array, axis=0)

        model = tf.keras.models.load_model("model.h5")

        predictions = model.predict(img_array)
        cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        fig, ax = plt.subplots()
        y_pos = np.arange(len(cifar10_labels))
        ax.barh(y_pos, predictions[0], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cifar10_labels)
        ax.invert_yaxis()
        ax.set_xlabel('Porbability')
        ax.set_title('CIFAR - Prediction')
        

        st.pyplot(fig)


        
    else:
        st.write("Please upload an image.")


if __name__ == "__main__":
    main()