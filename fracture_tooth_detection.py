import streamlit as st
import os
import PIL
import cv2
import secrets
import pymongo
import io
import cv2
from mongoengine import Document
from mongoengine import FileField
from mongoengine import *

connect(host="mongodb+srv://adithya_admin:YlV1pML2mjkP2LMb@dental-diagnosis.xevq3.mongodb.net/dental_diagnosis?retryWrites=true&w=majority")

class FractureDetector(Document):
    # by default collection_name is 'fs'
    # you can use collection_name parameter to set the name of the collection
    # use collection_name like this: FileField(collection_name='images')
    _id = StringField(required=True)
    description = StringField(required=True)
    image_fracture = FileField()


def write():
  st.set_option('deprecation.showfileUploaderEncoding', False)
  st.title("Fracture Tooth Detection By A.Adithya Sherwood IX-E")
  st.subheader('Disclaimer: Please check with your local specialized dentist, if you are in doubt please try atleast twice.')
  uploaded_file = st.file_uploader("Choose an image", type="jpg")
  conf_score = st.slider('Please Choose A Confidence Value',0.1,1.0,0.05)
  if uploaded_file is not None:
      file_random = secrets.token_hex(4)
      image = PIL.Image.open(uploaded_file)
      image = image.resize((512,512))
      image.save(f'./Test_Fracture{file_random}.jpg')

      #image_frac = open(f'./Test_Fracture{file_random}.jpg','rb')
      #fracture_form = FractureDetector(_id = secrets.token_hex(4),description='Uploaded Fracture Image',image_fracture=image_frac)
      #fracture_form.save()

      st.image(image, caption='Uploaded Image.', use_column_width=True)
      st.write("")
      os.system(f"python3 detect.py --weights './weights/best (2).pt' --img 512 --conf {str(conf_score)} --source ./Test_Fracture{file_random}.jpg --output ./inference/output ")
      image_pred = PIL.Image.open(f'./inference/output/Test_Fracture{file_random}.jpg')

      #image_frac_output = open(f'./inference/output/Test_Fracture{file_random}.jpg','rb')
      #fracture_form_out = FractureDetector(_id = secrets.token_hex(4),description='Predicted Fracture Image',image_fracture=image_frac_output)
      #fracture_form_out.save()

      st.image(image_pred, caption='Predictions.', use_column_width=True)
