# import streamlit as st

# import cv2

# st.title('🌱Plant disease prediction App🌿')

# uploaded_img = st.file_uploader("Choose a file",type=['jpg','jpeg','png'])

# if uploaded_img is not None:
#      img=cv2.imread(uploaded_img)
#      col1,col2=st.columns(2) 

#      with col1:
#           image=img.resize(150,150)
#           st.image(image)

      
# import streamlit as st
# import cv2
# import numpy as np

# st.title('Plant disease prediction App')

# uploaded_img = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

# if uploaded_img is not None:
#     # Convert the uploaded image to a NumPy array
#     file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)

#     # Decode the byte array into an image using OpenCV
#     img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

#     # Resize the image (optional)
#     resized_img = cv2.resize(img, (150, 150))  # You can adjust these dimensions

#     col1, col2 = st.columns(2)

#     with col1:
#         st.image(resized_img)  # Display the image

import json
import cv2
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

class_indices=json.load(open('class_indices.json'))
model=load_model('plant_model.h5')

def prediction(model,image_path,class_indices):
#     img=cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
    # Convert byte array to NumPy array and decode as image
    img = cv2.imdecode(np.frombuffer(uploaded_img.read(), np.uint8), cv2.IMREAD_UNCHANGED)
 
    img_resized=cv2.resize(img,(224,224))
    img=np.expand_dims(img_resized,axis=0)
    pred=model.predict(img)
    predictions=class_indices[str(np.argmax(pred))]
    return predictions
# image_path ='plantvillage dataset/plant_data/test/Corn_(maize)___Common_rust_/RS_Rust 1564.JPG'
image_path='plantvillage dataset/plant_data/test/Grape___Black_rot/00cab05d-e87b-4cf6-87d8-284f3ec99626___FAM_B.Rot 3244.JPG'

st.title('🌱Plant disease prediction App🌿')

uploaded_img = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])
# img='0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG'
if uploaded_img is not None:

    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded_img)
    with col2:
       
        if st.button('Classify'):
            st.success('correct prediction')
            predictions=prediction(model,uploaded_img,class_indices)
            st.success(f'prediction:{predictions}')