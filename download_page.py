import streamlit as st
import os
import PIL
import cv2
import secrets
import streamlit.components.v1 as stc



def write():
    st.title("Made By A.Adithya Sherwood IX-E")
    st.subheader('Available In Android,Windows,MacOS')
    st.subheader('Linux Comming Soon!')
    stc.html("""<iframe frameborder="0" src="https://itch.io/embed/807153?linkback=true&amp;border_width=5" width="560" height="175"><a href="https://the-ml-hero.itch.io/dental-diagnosis">Dental Diagnosis by The-ML-Hero</a></iframe>""")
    st.warning("I Recommend That You Use The WebApp Version Of This Project Since You Get Almost All Features Of the Native Build")
