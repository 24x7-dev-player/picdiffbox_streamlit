import streamlit as st
import cv2
import numpy as np
from helper import *
from PIL import Image, ImageOps
import io



def main():
    st.title("Image Difference Highlighter")
    
    st.markdown("<p style='color:Yellow'>Deep Learning Models :- </p>", unsafe_allow_html=True)
    
    st.text("💠 Siamese Network")
    st.text("💠 Euclidean Distance and Thresholding")
    st.text("💠 Bounding Box Prediction Network")
    
    st.markdown("<p style='color:skyblue'>Deep learning model can do this but not working well because of lack of trainable data & this task is like comparision. (A needle doesn't need a sword)</p>", unsafe_allow_html=True)
    
    st.markdown("<p style='color:Yellow'>Computer Vision Techniques :- </p>", unsafe_allow_html=True)
    st.text("💠 GIF VISUALISATION")
    st.text("💠 Approachs1 :- MASKING")
    st.text("💠 Approachs2 :- Structural Similarity Index (SSI)")
    st.text("💠 Approachs3.1 :- Absolute Diffrence")
    st.text("💠 Approachs3.2 :- Absolute Diffrence with contours")
    st.text("💠 Approachs4 :- Pixel Wise Comparision")
    
    
    st.markdown("<p style='color:Yellow'>Quantitative metrics & Evaluation metrics :- </p>", unsafe_allow_html=True)
    st.text("💠 Structural Similarity Index Measure(SSIM)")
    st.text("💠 Similarity Score (MSE)")
    st.text("💠 Histogram Difference")
    st.text("💠 Peak signal-to-noise ratio")
    
    st.text("💠 Number of changes")
    st.text("💠 Total change")
    st.text("💠 Average change per change region")
    st.text("💠 Max change in a change region")

    # allow user to upload images
    uploaded_file1 = st.file_uploader("Upload first image", type=["png", "jpg", "jpeg"])
    uploaded_file2 = st.file_uploader("Upload second image", type=["png", "jpg", "jpeg"])
    
    
            
    if uploaded_file1 and uploaded_file2:
        image1 = cv2.imdecode(np.frombuffer(uploaded_file1.read(), np.uint8), cv2.IMREAD_COLOR)
        image2 = cv2.imdecode(np.frombuffer(uploaded_file2.read(), np.uint8), cv2.IMREAD_COLOR)
        
        gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        
        imagea1 = Image.open(uploaded_file1)
        imagea2 = Image.open(uploaded_file2)
        
     
        if uploaded_file1 is not None and uploaded_file2 is not None:
            gif_path = create_gif_from_images(imagea1, imagea2, duration=100)

        st.header(" GIF Visualization")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(uploaded_file1, caption='Uploaded File 1', width=200)
        with col2:
            st.image(uploaded_file2, caption='Uploaded File2', width=200)
        with col3:
            display_gif(gif_path)
     
     
     
        st.header("Approach 1:- MASKING")
        image1_bytes, image2_bytes, diff_bytes = masking(image1, image2)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image1_bytes, caption='Image 1', width=200)
        with col2:
            st.image(image2_bytes, caption='Image 2', width=200)
        with col3:
            st.image(diff_bytes, caption='Difference Image', width=200)
            
            
            
        st.header("Approach 2:- Structural Similarity Index (SSIM)")
        st.text("Disadvantage:- Both file must be same size.")
        image1, image2, diff, diff_box, mask, filled_after = Structural_Similarity_Index(image1, image2)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image1, caption='Image1', width=200)
        with col2:
            st.image(image2, caption='Image 2', width=200)
        with col3:
            st.image(filled_after, caption='Filled After', width=200)
            
            
            
        st.header("Approach 3.1   :- Absolute Difference")
        image1, image2, diff = absolute_diffrence(image1, image2)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image1, caption='Image1', width=200)
        with col2:
            st.image(image2, caption='Image 2', width=200)
        with col3:
            st.image(diff, caption='Absolute Diffrence', width=200)
            
            
        st.header("Approach 3.2   :- Absolute Diff. with contours")
        result_image, diff_image, threshold_image  = absolute_diffrence2(gray_image1,gray_image2)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(result_image, channels="RGB", caption="Images with Bounding Boxes")
        with col2:
            st.image(diff_image, channels="GRAY", caption="Difference Image")
        with col3:
            st.image(threshold_image, channels="GRAY", caption="Thresholded Difference Image")
            
            
        st.header("Approach 4   :- Pixel Wise Comparision")
        image1, difference  = pixelwise_comparison(imagea1, imagea2)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(imagea1, channels="RGB", caption="Images1")
        with col2:
            st.image(imagea2, channels="GRAY", caption="Image2")
        with col3:
            st.image(image1, channels="GRAY", caption="Difference Image")

        
        
        
        st.header("Quantitative metrics & Evaluation metrics")
        ssim_score, mse_score, psnr_score, difference1 = calculate_metrics(imagea1, imagea2)
        result_image, diff, evolution_matrix, num_changes, total_change, average_change, max_change = calculate_metrics2(imagea1, imagea2)

        st.text(f"Structural Similarity Index Measure(SSIM: {ssim_score}")
        st.text(f"Mean Square Error(gray): {mse_score}")
        st.text(f"Peak signal-to-noise ratio: {psnr_score}")
        st.text(f"Histogram Difference : {difference1}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(result_image, caption="Original Image with Bounding Boxes", use_column_width=True)
        with col2:
            st.image(diff, caption="Difference Image", use_column_width=True)
        with col3:
            st.image(evolution_matrix, caption="Evolution Matrix", use_column_width=True)

        
        # st.image(result_image, caption="Original Image with Bounding Boxes", use_column_width=True)
        # st.image(diff, caption="Difference Image", use_column_width=True)
        # st.image(evolution_matrix, caption="Evolution Matrix", use_column_width=True)
        
        
        st.write("Number of changes:", num_changes)
        st.write("Total change:", total_change)
        st.write("Average change per change region:", average_change)
        st.write("Max change in a change region:", max_change)
            



if __name__ == "__main__":
    main()

