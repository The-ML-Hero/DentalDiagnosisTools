import streamlit as st
import os
import PIL
import streamlit.components.v1 as stc


def write():
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.title("Flurosis Tooth Detection By A.Adithya Sherwood IX-E")
    st.subheader('Disclaimer: Please check with your local specialized dentist, if you are in doubt please try atleast twice.')
    conf_score = st.slider('Please Choose A Confidence Value',0.1,1.0,0.05)
    uploaded_file = st.file_uploader("Choose an image", type="jpg")    

    
    if uploaded_file is not None:
        image = PIL.Image.open(uploaded_file)
        image = image.resize((416,416))
        image.save(f'./Test_Flurosis.jpg')
        image_flurosis = open(f'./Test_Flurosis.jpg','rb')
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        os.system(f"python3 detect.py --weights './weights/best (2).pt' --img 416 --conf {str(conf_score)} --source ./Test_Flurosis.jpg --output ./test.jpg")
        image_pred = PIL.Image.open(f'./test.jpg')
        st.image(image_pred, caption='Predictions.', use_column_width=True)
