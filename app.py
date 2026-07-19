import streamlit as st 
import numpy as np 
from PIL import Image 
from tensorflow.keras.models import load_model 
from streamlit_cropper import st_cropper 

st.set_page_config(page_title="Face Mask Detector", layout="centered")
st.title("😷 Face Mask Detection Project") 

model = load_model("face-mask-detector.keras") 

option = st.selectbox("Select Input Method", ["Image", "Capture"]) 

if option == "Image": 
    camera_image = None 
    uploaded_file = st.file_uploader("Upload image", type = ["jpg", "jpeg", "png"]) 
else: 
    camera_image = st.camera_input("Capture a photo") 
    uploaded_file = None 

image_source = uploaded_file if option == "Image" else camera_image

if image_source: 
    image = Image.open(image_source) 
    st.write("---") 
    st.subheader("Crop the face area:") 
    
    cropped_img = st_cropper(image, realtime_update=True, box_color='red', aspect_ratio=None) 
    
    cropped_img = cropped_img.resize((150, 150)) 
    st.image(cropped_img, caption="Preview of Cropped Image") 

st.write("---")

if st.button("Detect Mask"): 
    image_to_detect = None 
    
    if uploaded_file is not None: 
        image_to_detect = cropped_img 
    elif camera_image is not None: 
        image_to_detect = cropped_img 
    else: 
        st.error("Please upload or capture an image first!") 
        
    if image_to_detect is not None: 
        image_to_detect = image_to_detect.convert("RGB") 
        img = image_to_detect.resize((150, 150)) 
        img_array = np.array(img, dtype=np.float32) / 255.0 
        img_array = np.expand_dims(img_array, axis=0) 
        
        with st.spinner("Analyzing image..."):
            result = model.predict(img_array) 
            
        st.write("### Result:")
        if result[0,0] <= 0.5: 
            st.success("✅ Safe: Person is wearing a mask!") 
        else: 
            st.error("🚨 Warning: Person is NOT wearing a mask!")
