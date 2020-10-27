import cv2
import PIL
import streamlit as st
import matplotlib.pyplot as plt


def write():
  st.set_option('deprecation.showfileUploaderEncoding', False)
  st.title('Tooth Fracture Regeneration by A.Adithya Sherwood IX-E')
  st.subheader('Disclaimer: Please check with your local specialized dentist, if you are in doubt please try atleast twice.')
  uploaded_file_img = st.file_uploader("Choose an Input Image", type="png")
  uploaded_file_mask = st.file_uploader("Choose an Mask Image", type="png")
  radius = st.slider('Radius', 1.0, 416.0,0.5 )
  flags = st.selectbox("Please Select a Model",(cv2.INPAINT_TELEA,cv2.INPAINT_NS))
  if flags == 0:
    st.write("You Selected Telea Model")
  else:
    st.write("You Selected Navier Stroke Model")  
  if uploaded_file_img and uploaded_file_mask is not None:
      img_pil = PIL.Image.open(uploaded_file_img)
      mask_pil = PIL.Image.open(uploaded_file_mask)
      img_pil = img_pil.resize((416,416))
      mask_pil = mask_pil.resize((416,416))
      img_pil.save('./input.png')
      mask_pil.save('./mask.png')
      st.image(img_pil, caption='Uploaded Image.', use_column_width=True)
      st.write("")
      st.image(mask_pil, caption='Uploaded Mask.', use_column_width=True)
      image = cv2.imread('./input.png')
      mask = cv2.imread('./mask.png')
      mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
      output = cv2.inpaint(image, mask,inpaintRadius=radius, flags=flags)
      cv2.imwrite('./output.png',output)
      output = plt.imread('./output.png')
      st.image(output, caption='Generated Image.', use_column_width=True)