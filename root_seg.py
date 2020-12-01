import streamlit as st
from streamlit_cropper import st_cropper
import requests
import cv2
import wget
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import PIL
import numpy as np
from detectron2.utils.visualizer import ColorMode
import os
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog,DatasetCatalog
import numpy as np
from mongoengine import Document
from mongoengine import FileField
from mongoengine import *
import secrets

connect(host="mongodb+srv://adithya_admin:YlV1pML2mjkP2LMb@dental-diagnosis.xevq3.mongodb.net/dental_diagnosis?retryWrites=true&w=majority")

class RootDetector(Document):
    # by default collection_name is 'fs'
    # you can use collection_name parameter to set the name of the collection
    # use collection_name like this: FileField(collection_name='images')
    _id = StringField(required=True)
    description = StringField(required=True)
    image_root = FileField()
    

def write():
    
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.title('Root Anatomy Prediction by A.Adithya Sherwood IX-E')
    st.subheader('Disclaimer: Please check with your local specialized dentist, if you are in doubt please try atleast twice.')
    confidence =  st.slider(
    'Please Select a Confidence Value',
    0.1, 1.0, 0.05
    )
    uploaded_file_img = st.file_uploader("Choose an Input Image", type="png",accept_multiple_files=False)
    crop_image = st.checkbox('Crop?')
    if uploaded_file_img is not None:
        file_random = secrets.token_hex(4)
        o = int(np.random.randint(low=10301319,high=9987869996)) 
        cfg = get_cfg()
        cfg.MODEL.DEVICE='cpu'
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml"))
        register_coco_instances(f"tooth_segmentation_maskrcnn{o}",{},str(f"./annotations/instances_default.json"),str(f"./images"))
        cfg.DATASETS.TRAIN = (f"tooth_segmentation_maskrcnn{o}",)
        cfg.DATASETS.TEST = ()
        cfg.DATASETS.NUM_WORKERS = 2
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 100   # faster, and good enough for this toy dataset (default: 512)
        cfg.SOLVER.BASE_LR = 0.00025
        cfg.SOLVER.MAX_ITER = 800
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        cfg.MODEL.WEIGHTS = './MASK_RCNN_ROOT_SEGMENTATION.pth'  # path to the model we just trained
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence   # set a custom testing threshold
        predictor = DefaultPredictor(cfg)
        MetadataCatalog.get(f"tooth_segmentation_maskrcnn{o}").thing_classes = ["CShaped", "Normal"]
        img_pil = PIL.Image.open(uploaded_file_img)
        img_pil = img_pil.resize((512,512))
        img_pil.save(f'input_mask{file_random}.png')

        image_root_seg = open(f'input_mask{file_random}.png','rb')
        root_form = RootDetector(_id = secrets.token_hex(4),description='Uploaded Root Detector Image',image_root=image_root_seg)
        root_form.save()

        if crop_image True:
            st.image(img_pil, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            cropped_img = st_cropper(img_pil, realtime_update=True, box_color="#0000ff",
                                    aspect_ratio=(1,1))
            cropped_img.save(f'input_mask_cropped{file_random}.png')
            st.write("")
            st.image(cropped_img,caption='Cropped Image.', use_column_width=True)
            image = cv2.imread(f'input_mask_cropped{file_random}.png')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #image = cv2.copyMakeBorder(image, 150, 150, 150, 150, cv2.BORDER_CONSTANT,value=[255,255,255])
            output = predictor(image)
            v = Visualizer(image[:, :, ::-1],
                            metadata=MetadataCatalog.get(f"tooth_segmentation_maskrcnn{o}"), 
                            scale=2, 
                            # remove the colors of unsegmented pixels. This option is only available for segmentation models
            )

            out = v.draw_instance_predictions(output["instances"].to("cpu"))
            cv2.imwrite(f'output_MASK{file_random}.png',out.get_image()[:, :, ::-1])
            out_i = PIL.Image.open(f'output_MASK{file_random}.png') 
            st.image(out_i, caption='Predicted Masks', use_column_width=True)
            image_root_seg_out = open(f'output_MASK{file_random}.png','rb')
            root_form_out = RootDetector(_id = secrets.token_hex(4),description='Predicted Root Detector Image',image_root=image_root_seg_out)
            root_form_out.save()
         else:
            st.image(img_pil, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            cropped_img = st_cropper(img_pil, realtime_update=True, box_color="#0000ff",
                                    aspect_ratio=None)
            cropped_img.save(f'input_mask_cropped{file_random}.png')
            st.write("")
            st.image(cropped_img,caption='Cropped Image.', use_column_width=True)
            image = cv2.imread(f'input_mask_cropped{file_random}.png')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #image = cv2.copyMakeBorder(image, 150, 150, 150, 150, cv2.BORDER_CONSTANT,value=[255,255,255])
            output = predictor(image)
            v = Visualizer(image[:, :, ::-1],
                            metadata=MetadataCatalog.get(f"tooth_segmentation_maskrcnn{o}"), 
                            scale=2, 
                            # remove the colors of unsegmented pixels. This option is only available for segmentation models
            )

            out = v.draw_instance_predictions(output["instances"].to("cpu"))
            cv2.imwrite(f'output_MASK{file_random}.png',out.get_image()[:, :, ::-1])
            out_i = PIL.Image.open(f'output_MASK{file_random}.png') 
            st.image(out_i, caption='Predicted Masks', use_column_width=True)
            image_root_seg_out = open(f'output_MASK{file_random}.png','rb')
            root_form_out = RootDetector(_id = secrets.token_hex(4),description='Predicted Root Detector Image',image_root=image_root_seg_out)
            root_form_out.save()
"""def write():
  st.set_option('deprecation.showfileUploaderEncoding', False)
  st.title('Root Anatomy Prediction by A.Adithya Sherwood IX-E')
  st.subheader('Disclaimer: Please check with your local specialized dentist, if you are in doubt please try atleast twice.')
  confidence =  st.slider(
    'Please Select a Confidence Value',
    0.1, 1.0, 0.05
    )
  option = st.selectbox(
    'Please Select A Model',
     ('Mask RCNN v1', 'Mask RCNN v2'))
  add_border = st.checkbox('Add Border? Sometimes this helps in better performance',value=False)
  uploaded_file_img = st.file_uploader("Choose an Input Image", type="png",accept_multiple_files=False)
  if uploaded_file_img is not None:
    o = int(np.random.randint(low=10301319,high=9987869996)) 
    cfg = get_cfg()
    cfg.MODEL.DEVICE='cpu'
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
    register_coco_instances(f"tooth_segmentation_maskrcnn{o}",{},str(f"./annotations/instances_default.json"),str(f"./images"))
    cfg.DATASETS.TRAIN = (f"tooth_segmentation_maskrcnn{o}",)
    cfg.DATASETS.TEST = ()
    cfg.DATASETS.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 350
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.WEIGHTS = './old mask rcnn.pth'  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    MetadataCatalog.get(f"tooth_segmentation_maskrcnn{o}").thing_classes = ["CShaped", "Normal"]
    if add_border == True and option == 'Mask RCNN v1':
      
      cfg.MODEL.WEIGHTS = './old mask rcnn.pth'
      img_pil = PIL.Image.open(uploaded_file_img)
      img_pil = img_pil.resize((512,512))
      img_pil.save('input.png')
      st.image(img_pil, caption='Uploaded Image.', use_column_width=True)
      st.write("")
      image = cv2.imread('input.png')
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image = cv2.copyMakeBorder(image, 150, 150, 150, 150, cv2.BORDER_CONSTANT,value=[255,255,255])
      output = predictor(image)
      v = Visualizer(image[:, :, ::-1],
                    metadata=MetadataCatalog.get(f"tooth_segmentation_maskrcnn{o}"), 
                    scale=2, 
                    # remove the colors of unsegmented pixels. This option is only available for segmentation models
      )

      out = v.draw_instance_predictions(output["instances"].to("cpu"))
      cv2.imwrite('output.png',out.get_image()[:, :, ::-1])
      out_i = PIL.Image.open('output.png') 
      st.image(out_i, caption='Predicted Masks', use_column_width=True)

    elif add_border == False and option == 'Mask RCNN v1':
      cfg.MODEL.WEIGHTS = './old mask rcnn.pth'
      img_pil = PIL.Image.open(uploaded_file_img)
      img_pil = img_pil.resize((512,512))
      img_pil.save('input.png')
      st.image(img_pil, caption='Uploaded Image.', use_column_width=True)
      st.write("")
      image = cv2.imread('input.png')
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      #image = cv2.copyMakeBorder(image, 150, 150, 150, 150, cv2.BORDER_CONSTANT,value=[255,255,255])
      output = predictor(image)
      v = Visualizer(image[:, :, ::-1],
                    metadata=MetadataCatalog.get(f"tooth_segmentation_maskrcnn{o}"), 
                    scale=2, 
                    # remove the colors of unsegmented pixels. This option is only available for segmentation models
      )

      out = v.draw_instance_predictions(output["instances"].to("cpu"))
      cv2.imwrite('output.png',out.get_image()[:, :, ::-1])
      out_i = PIL.Image.open('output.png') 
      st.image(out_i, caption='Predicted Masks', use_column_width=True)
    elif add_border == True and option == 'Mask RCNN v2':
      cfg.MODEL.WEIGHTS = './new mask rcnn.pth'
      img_pil = PIL.Image.open(uploaded_file_img)
      img_pil = img_pil.resize((512,512))
      img_pil.save('input.png')
      st.image(img_pil, caption='Uploaded Image.', use_column_width=True)
      st.write("")
      image = cv2.imread('input.png')
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image = cv2.copyMakeBorder(image, 150, 150, 150, 150, cv2.BORDER_CONSTANT,value=[255,255,255])
      output = predictor(image)
      v = Visualizer(image[:, :, ::-1],
                    metadata=MetadataCatalog.get(f"tooth_segmentation_maskrcnn{o}"), 
                    scale=2, 
                    # remove the colors of unsegmented pixels. This option is only available for segmentation models
      )

      out = v.draw_instance_predictions(output["instances"].to("cpu"))
      cv2.imwrite('output.png',out.get_image()[:, :, ::-1])
      out_i = PIL.Image.open('output.png') 
      st.image(out_i, caption='Predicted Masks', use_column_width=True)
    elif add_border == True and option == 'Mask RCNN v2':
      cfg.MODEL.WEIGHTS = './new mask rcnn.pth'
      img_pil = PIL.Image.open(uploaded_file_img)
      img_pil = img_pil.resize((512,512))
      img_pil.save('input.png')
      st.image(img_pil, caption='Uploaded Image.', use_column_width=True)
      st.write("")
      image = cv2.imread('input.png')
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image = cv2.copyMakeBorder(image, 150, 150, 150, 150, cv2.BORDER_CONSTANT,value=[255,255,255])
      output = predictor(image)
      v = Visualizer(image[:, :, ::-1],
                    metadata=MetadataCatalog.get(f"tooth_segmentation_maskrcnn{o}"), 
                    scale=2, 
                    # remove the colors of unsegmented pixels. This option is only available for segmentation models
      )

      out = v.draw_instance_predictions(output["instances"].to("cpu"))
      cv2.imwrite('output.png',out.get_image()[:, :, ::-1])
      out_i = PIL.Image.open('output.png') 
      st.image(out_i, caption='Predicted Masks', use_column_width=True)       

    elif add_border == True and option == 'Mask RCNN v2':
      cfg.MODEL.WEIGHTS = './new mask rcnn.pth'
      img_pil = PIL.Image.open(uploaded_file_img)
      img_pil = img_pil.resize((512,512))
      img_pil.save('input.png')
      st.image(img_pil, caption='Uploaded Image.', use_column_width=True)
      st.write("")
      image = cv2.imread('input.png')
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image = cv2.copyMakeBorder(image, 150, 150, 150, 150, cv2.BORDER_CONSTANT,value=[255,255,255])
      output = predictor(image)
      v = Visualizer(image[:, :, ::-1],
                    metadata=MetadataCatalog.get(f"tooth_segmentation_maskrcnn{o}"), 
                    scale=2, 
                    # remove the colors of unsegmented pixels. This option is only available for segmentation models
      )

      out = v.draw_instance_predictions(output["instances"].to("cpu"))
      cv2.imwrite('output.png',out.get_image()[:, :, ::-1])
      out_i = PIL.Image.open('output.png') 
      st.image(out_i, caption='Predicted Masks', use_column_width=True)
    elif add_border == False and option == 'Mask RCNN v2':
      cfg.MODEL.WEIGHTS = './new mask rcnn.pth'
      img_pil = PIL.Image.open(uploaded_file_img)
      img_pil = img_pil.resize((512,512))
      img_pil.save('input.png')
      st.image(img_pil, caption='Uploaded Image.', use_column_width=True)
      st.write("")
      image = cv2.imread('input.png')
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      #image = cv2.copyMakeBorder(image, 150, 150, 150, 150, cv2.BORDER_CONSTANT,value=[255,255,255])
      output = predictor(image)
      v = Visualizer(image[:, :, ::-1],
                    metadata=MetadataCatalog.get(f"tooth_segmentation_maskrcnn{o}"), 
                    scale=2, 
                    # remove the colors of unsegmented pixels. This option is only available for segmentation models
      )

      out = v.draw_instance_predictions(output["instances"].to("cpu"))
      cv2.imwrite('output.png',out.get_image()[:, :, ::-1])
      out_i = PIL.Image.open('output.png') 
      st.image(out_i, caption='Predicted Masks', use_column_width=True)   
"""



