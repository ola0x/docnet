import cv2
import numpy as np 
import streamlit as st
from auto_doc_p import load_model, Model, process_img_path, find_similar_images
from util import process_img, load_images, process_img_path

# init
model = load_model()

image_folder = "demo-f"

if __name__ == "__main__":

    html_temp = """
        <div style="background-color:black;padding:5px">
        <h2 style="color:white;text-align:center;">Demo AI Document verification</h2>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True)
    image_files, manuscripts = load_images(image_folder)
    view_manuscripts = st.multiselect("Select Document to verify", manuscripts)
    n = st.number_input("Select Grid Width", 1, 5, 3)

    view_images = []
    for image_file in image_files:
        if any(manuscript in image_file for manuscript in view_manuscripts):
            view_images.append(image_file)
    groups = []
    for i in range(0, len(view_images), n):
        groups.append(view_images[i:i+n])

    for group in groups:
        cols = st.columns(n)
        for i, image_file in enumerate(group):
            cols[i].image(image_file)
    
    db_paths = groups
    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    
    db_representations = []
    
    for folder_images in db_paths:
        folder_representations = []
        for image_path in folder_images:
            img_representation = vgg_face_descriptor.predict(process_img_path(image_path))[0, :]
            folder_representations.append(img_representation)
        db_representations.append(folder_representations)

    # print(groups)
    image_upload = st.file_uploader(
        "Upload Document Image", type=["png", "jpg", "jpeg"])
    
    if image_upload is not None:

        # To See details
        file_details = {"filename": image_upload.name, "filetype": image_upload.type,
            "filesize": image_upload.size}

            # Display the predict button just when an image is being uploaded
        if not image_upload:
            st.warning("Please upload an INVOICE image before proceeding!")
            st.stop()
        else:
                # image_as_bytes = uploaded_file.read()
            image_as_bytes = np.asarray(
                bytearray(image_upload.read()), dtype=np.uint8)
            img = cv2.imdecode(image_as_bytes, 1)
            img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            st.image(img_cv)
            pred_button = st.button("Verify")
        
        if pred_button:
            img1_representation = vgg_face_descriptor.predict(process_img(image_as_bytes))[0, :]
            similar_indices = []
    
            for folder_index, folder_representations in enumerate(db_representations):
                similar_indices_folder = find_similar_images(img1_representation, folder_representations)
                similar_indices.extend([(folder_index, idx) for idx in similar_indices_folder])
            
            if len(similar_indices) > 0:
                st.text("The new doc is verified")
            else:
                st.text("The doc is not verified")
