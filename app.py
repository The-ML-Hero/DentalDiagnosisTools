import os
import requests
import shutil
import sys
import streamlit as st
import streamlit.components.v1 as stc
import tooth_reconstruction
import display_app as udisp
#import Pulp_Segmentation
import root_seg
import fracture_tooth_detection
import flurosis_tooth_detection
import download_page

if not os.path.exists("./MASK_RCNN_ROOT_SEGMENTATION"):
    os.system("git clone https://gitlab.com/sherwoodadithya/dentaldiagnosistoolkitai.git")
    shutil.move('./dentaldiagnosistoolkitai/MASK_RCNN_ROOT_SEGMENTATION.pth', './MASK_RCNN_ROOT_SEGMENTATION.pth')

#
#
#os.system("ls")
#os.system("ifconfig")
MENU = {
    
    "Fracture Tooth Detection" : fracture_tooth_detection,
    "Root Anatomy Prediction" : root_seg,
    "Flurosis Detection" : flurosis_tooth_detection,	
    "Tooth Fracture Reconstruction" : tooth_reconstruction,	
    "Download":download_page
}

st.sidebar.title("Choose A Use Case")

menu_selection = st.sidebar.radio("Use Case", list(MENU.keys()))

menu = MENU[menu_selection]


with st.spinner(f"Loading {menu_selection} ..."):
      udisp.render_page(menu)

