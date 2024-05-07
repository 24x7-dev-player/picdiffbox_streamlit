import cv2
import numpy as np
from skimage.metrics import structural_similarity
import streamlit as st
import io
import base64
from PIL import Image
import os

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr


def masking(image1, image2):    
    # compute difference
    difference = cv2.subtract(image1, image2)

    # color the mask red
    Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    difference[mask != 255] = [0, 0, 255]

    # add the red mask to the images to make the differences obvious
    image1[mask != 255] = [0, 0, 255]
    image2[mask != 255] = [0, 0, 255]

    # convert images back to bytes
    image1_bytes = cv2.imencode('.png', image1)[1].tobytes()
    image2_bytes = cv2.imencode('.png', image2)[1].tobytes()
    diff_bytes = cv2.imencode('.png', difference)[1].tobytes()

    return image1_bytes, image2_bytes, diff_bytes


def Structural_Similarity_Index(image1, image2):  
    
    before_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Compute SSIM between the two images
    (score, diff) = structural_similarity(before_gray, after_gray, full=True)
    st.text("Image Similarity: {:.4f}%".format(score * 100))

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1] 
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")
    diff_box = cv2.merge([diff, diff, diff])

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(image1.shape, dtype='uint8')
    filled_after = image2.copy()

    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(image1, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.rectangle(image2, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.drawContours(mask, [c], 0, (255,255,255), -1)
            cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

    return image1, image2, diff, diff_box, mask, filled_after
 

def absolute_diffrence(image1, image2):    

    def mse(image1, image2):
        diff = image1.astype("float") - image2.astype("float")
        squared_diff = diff ** 2
        mse = squared_diff.mean()
        return mse    
    
    similarity_score = mse(image1, image2)  
    st.text(f"Similarity Score (MSE): {similarity_score}")
    
    
    diff = 255 - cv2.absdiff(image1, image2)        
    # convert images back to bytes
    image1_bytes = cv2.imencode('.png', image1)[1].tobytes()
    image2_bytes = cv2.imencode('.png', image2)[1].tobytes()

    return image1_bytes, image2_bytes, diff


def absolute_diffrence2(image1, image2):
    
     # Compute absolute difference between the two images
    diff = cv2.absdiff(image1, image2)
    
    # Threshold the difference image
    threshold = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)[1]
    
    # Find contours of the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw bounding boxes around the contours
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    
    result_image = cv2.imencode('.png', image1)[1].tobytes()
    

    # Calculate similarity score (percentage of pixels that are different)
    total_pixels = np.prod(image1.shape)
    different_pixels = np.count_nonzero(diff)
    similarity_score = (1 - (different_pixels / total_pixels)) * 100
    return result_image, diff, threshold


def create_gif_from_images(image1, image2, duration=100):
    # Resize images
    frames = [image1, image2]
    gif_path = 'temp.gif'
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    return gif_path

def display_gif(gif_path):
    st.image(gif_path, caption='Animation', use_column_width=True, width=200)
    
    
def pixelwise_comparison(image1, image2):
    # Compute the absolute difference between the images
    image1 = np.array(image1)
    image2 = np.array(image2)
    
    
    # # Calculate the histograms
    hist1 = cv2.calcHist([image1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([image2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    # # Calculate the Chi-Squared distance
    difference1 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    st.text(f"Histogram Difference : {difference1}")
    
    
    difference = cv2.absdiff(image1, image2)

    # Convert the difference image to grayscale
    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to get a binary mask of the differences
    _, threshold = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around the contours
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image1, difference


def calculate_metrics(image1, image2):
    
    image1 = np.array(image1)
    image2 = np.array(image2)
    
    
    def mse(image1, image2):
        diff = image1.astype("float") - image2.astype("float")
        squared_diff = diff ** 2
        mse = squared_diff.mean()
        return mse    
    
    similarity_score = mse(image1, image2)  
    st.text(f"Similarity Score (MSE): {similarity_score}")
    
    
    # Calculate the histograms
    hist1 = cv2.calcHist([image1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([image2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    # # Calculate the Chi-Squared distance
    difference1 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    
    # Convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculate SSIM
    ssim_score, _ = ssim(gray1, gray2, full=True)

    # Calculate MSE
    mse_score = mean_squared_error(gray1, gray2)

    # Calculate PSNR
    psnr_score = psnr(gray1, gray2)

    return ssim_score, mse_score, psnr_score, difference1


def calculate_metrics2(image1, image2):
    
    image1 = np.array(image1)
    image2 = np.array(image2)
    
    # Convert images to grayscale
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute the Structural Similarity Index (SSI) between the two images
    (score, diff) = ssim(gray_image1, gray_image2, full=True)
    diff = (diff * 255).astype("uint8")
    # Threshold the difference image, find contours to localize differences
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize evolution matrix and metrics
    evolution_matrix = np.zeros(gray_image1.shape)
    num_changes = 0
    total_change = 0

    # Draw bounding boxes around the differences and update metrics
    for i, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Draw number in the bounding box
        cv2.putText(image1, str(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        evolution_matrix[y:y+h, x:x+w] += 1
        num_changes += 1
        total_change += cv2.countNonZero(thresh[y:y+h, x:x+w])

    # Calculate quantitative metrics
    average_change = total_change / num_changes if num_changes > 0 else 0
    max_change = np.max(evolution_matrix)
    
    evolution_matrix = cv2.normalize(evolution_matrix, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    
    
    
    return image1, diff, evolution_matrix, num_changes, total_change, average_change, max_change


