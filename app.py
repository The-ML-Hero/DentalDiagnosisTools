import os
import requests
import sys
import streamlit as st
import streamlit.components.v1 as stc
import tooth_reconstruction
import display_app as udisp
#import Pulp_Segmentation
import root_seg
import fracture_tooth_detection
import flurosis_tooth_detection

url_mask = 'https://blob-ap-south-1-ukyez4.s3.ap-south-1.amazonaws.com/sara/c6/c668/c668fa65-a0ec-4905-9d9a-341368dc3bb9.bin?response-content-disposition=attachment%3B%20filename%3D%22MASK_RCNN_ROOT_SEGMENTATION.pth%22&response-content-type=&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAI75SICYCOZ7DPWTA%2F20201027%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20201027T125742Z&X-Amz-SignedHeaders=host&X-Amz-Expires=1800&X-Amz-Signature=2592ecfbd1f7de226c77b15782d1f82a852115fe6602e80b1ddbdd8406bce030'
requests.get(url_mask ,stream=True)

MENU = {
    
    "Fracture Tooth Detection" : fracture_tooth_detection,
    "Root Anatomy Prediction" : root_seg,
    "Flurosis Detection" : flurosis_tooth_detection,	
    "Tooth Fracture Reconstruction" : tooth_reconstruction,	
}
st.sidebar.title("Choose A Use Case")
menu_selection = st.sidebar.radio("Use Case", list(MENU.keys()))
menu = MENU[menu_selection]

with st.spinner(f"Loading {menu_selection} ..."):
      udisp.render_page(menu)

