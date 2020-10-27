import streamlit as st
import os
import PIL
import cv2
import secrets
import streamlit.components.v1 as stc
from mongoengine import Document
from mongoengine import FileField
from mongoengine import *

connect(host="mongodb+srv://adithya_admin:YlV1pML2mjkP2LMb@dental-diagnosis.xevq3.mongodb.net/dental_diagnosis?retryWrites=true&w=majority")

class FlurosisDetector(Document):
    # by default collection_name is 'fs'
    # you can use collection_name parameter to set the name of the collection
    # use collection_name like this: FileField(collection_name='images')
    _id = StringField(required=True)
    description = StringField(required=True)
    image_fluro = FileField()

def write():
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.title("Flurosis Tooth Detection By A.Adithya Sherwood IX-E")
    st.subheader('Disclaimer: Please check with your local specialized dentist, if you are in doubt please try atleast twice.')
    conf_score = st.slider('Please Choose A Confidence Value',0.1,1.0,0.05)
    uploaded_file = st.file_uploader("Choose an image", type="jpg")    

    
    if uploaded_file is not None:
        file_random = secrets.token_hex(4)
        image = PIL.Image.open(uploaded_file)
        image = image.resize((416,416))
        image.save(f'./Test_Flurosis{file_random}.jpg')

        #image_flurosis = open(f'./Test_Flurosis{file_random}.jpg','rb')
        #flurosis_form = FlurosisDetector(_id = secrets.token_hex(4),description='Uploaded Flurosis Image',image_fluro=image_flurosis)
        #flurosis_form.save()

        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        os.system(f"python3 detect.py --weights './weights/best (2).pt' --img 416 --conf {str(conf_score)} --source ./Test_Flurosis{file_random}.jpg --output ./inference/output")
        image_pred = PIL.Image.open(f'./inference/output/Test_Flurosis{file_random}.jpg')

        #image_flurosis_out = open(f'./inference/output/Test_Flurosis{file_random}.jpg','rb')
        #flurosis_form_out = FlurosisDetector(_id = secrets.token_hex(4),description='Predicted Flurosis Image',image_fluro=image_flurosis_out)
        #flurosis_form_out.save()

        st.image(image_pred, caption='Predictions.', use_column_width=True)
