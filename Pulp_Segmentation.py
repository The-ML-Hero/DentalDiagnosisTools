from fastai import *
from fastai.vision import *
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import PIL
import os
import sys
import cv2
learn = load_learner('./Unet-V2')

def write():
  st.title('Tooth Internal Anatomy Prediction by A.Adithya Sherwood IX-E')
  st.subheader('Disclaimer: Please check with your local specialized dentist, if you are in doubt please try atleast twice.')
  uploaded_file = st.file_uploader("Choose a Slice", type="png")
  if uploaded_file is not None:
      image_arr = plt.imread(uploaded_file)
      image = pil2tensor(image_arr,dtype=np.float32)
      st.image(image_arr, caption='Uploaded Image.', use_column_width=True)
      st.write("")
      cat,tensor,probs = learn.predict(Image(image))
      image_final = Image(tensor)
      image_final.save('./output.png')
      img_pred_mask = open_mask('./output.png')
      img_pred_mask.save('mask.png')
      img_pred = plt.imread('./mask.png')
      st.image(img_pred, caption='Predictions', use_column_width=True)
