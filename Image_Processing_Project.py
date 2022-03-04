                                           # Import Packages
import os
import cv2
import math
import webbrowser
import numpy as np
from tkinter import *
import tkinter.messagebox
import PIL.Image, PIL.ImageTk
from tkinter import filedialog
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import gaussian
from PIL import ImageTk, Image, ImageFilter
from skimage.segmentation import active_contour
from scipy.ndimage import maximum_filter, minimum_filter
from sklearn.cluster import MeanShift, estimate_bandwidth
from tkinter.filedialog import askopenfilename, asksaveasfilename
from itertools import chain

                                         # Programming Step
#os.chdir(r"G:\Image Proceesing_\filters\image\image (1)\image")

def Exit ():
    Exit=tkinter.messagebox.askyesno("Exit","Really you want to Exit ?!")
    if Exit >0:
        root.destroy()
        return

def Remove():  # remove noise
    img = cv2.imread(IMG, 0)
    kern = np.array([[1 / 9, 1 / 9, 1 / 9],
                     [1 / 9, 1 / 9, 1 / 9],
                     [1 / 9, 1 / 9, 1 / 9]])
    img = cv2.filter2D(img, -1, kern)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (350, 340), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()


def new_func():
    def save_file():
        fileName = QFileDialog.getOpenFileName(self,tr("Open Image"), "/home/jana", tr("Image Files (*.png *.jpg *.bmp)"))
        "Images (*.png *.xpm *.jpg);;Text files (*.txt);;XML files (*.xml)"
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setNameFilter(tr("Images (*.png *.xpm *.jpg)"))
        dialog.setViewMode(QFileDialog.Detail)
    return save_file

save_file = new_func()
#global current_img

def search_google():
    webbrowser.open("https://www.google.com/search?q=images")
    
def choose_image():  # select image
    global canvas, IMG, image, image_tk
    canvas = Canvas(root, width=350, height=340, bg='#E8CEA6')
    canvas.place(x=190, y=170)
    IMG = filedialog.askopenfilename()
    image = Image.open(IMG)
    image_tk = ImageTk.PhotoImage(image)
    image = image.resize((350, 340), Image.ANTIALIAS)
    image_tk = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    
# def choose_image():  # select image
#     global canvas, IMG, image, image_tk 
#     canvas = Canvas(root, width=350, height=340)
#     canvas.place(x=190, y=120)
#     IMG = filedialog.askopenfilename()
#     image = Image.open(IMG)
#     image_tk = ImageTk.PhotoImage(image)
#     image = image.resize((350, 340), Image.ANTIALIAS)
#     image_tk = ImageTk.PhotoImage(image)
#     canvas.create_image(0, 0, anchor=NW, image=image_tk)
    






#ideal low pass
def ideal_lowpass():
    img = cv2.imread(IMG,0)
    dft = np.fft.fft2(img, axes=(0,1))
    dft_shift = np.fft.fftshift(dft)
    # generate spectrum from magnitude image (for viewing only)
    mag = np.abs(dft_shift)
    spec = np.log(mag) / 20
    x,y=np.shape(img)
    midpointx,midpointy=x//2 ,y//2
    maskideal=np.zeros((x,y), np.uint8)
    Do=50
    for u in range(0,x):
        for v in range(0,y):
              if np.sqrt(((u-midpointx)**2)+((v-midpointy)**2))<=Do:
                  maskideal[u][v]=255

    pil_img=(dft_shift*maskideal)/255
    pil_img=np.fft.ifftshift(pil_img)
    pil_img=np.fft.ifft2(pil_img, axes=(0,1))
    pil_img=np.abs(pil_img).clip(0,255).astype(np.uint8)
    img = cv2.resize(pil_img, (350, 340), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()


 #butter worth

def Butterworth_lowpass():
    def im2double(im):
        max_val = np.max(im)
        min_val = np.min(im)
        return np.round((im.astype('float') - min_val) / (max_val - min_val) * 255)
    img = cv2.imread(IMG,0)
    dft = np.fft.fft2(img, axes=(0,1))
    dft_shift = np.fft.fftshift(dft)
    # generate spectrum from magnitude image (for viewing only)
    mag = np.abs(dft_shift)
    spec = np.log(mag) / 20
    x,y=np.shape(img)
    midpointx,midpointy=x//2 ,y//2
    maskbutter=np.zeros((x,y))
    Do=50
    for u in range(0,x):
        for v in range(0,y):
                  duv=np.sqrt(((u-midpointx)**2)+((v-midpointy)**2))
                  maskbutter[u][v]=1/(1+(duv/Do)**4)

    #cv2.imshow("mask",maskbutter)
    maskbutter=im2double(maskbutter)
    pil_img=(dft_shift*maskbutter)/255
    pil_img=np.fft.ifftshift(pil_img)
    pil_img=np.fft.ifft2(pil_img, axes=(0,1))
    pil_img=np.abs(pil_img).clip(0,255).astype(np.uint8)
    img = cv2.resize(pil_img, (350, 340), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()


# Gaussien low pass filter
def Gaussuien_lowpass():
    def im2double(im):
        max_val = np.max(im)
        min_val = np.min(im)
        return np.round((im.astype('float') - min_val) / (max_val - min_val) * 255)
    img = cv2.imread(IMG,0)
    dft = np.fft.fft2(img, axes=(0,1))
    dft_shift = np.fft.fftshift(dft)
    # generate spectrum from magnitude image (for viewing only)
    mag = np.abs(dft_shift)
    spec = np.log(mag) / 20
    x,y=np.shape(img)
    midpointx,midpointy=x//2 ,y//2
    maskgussine=np.zeros((x,y))
    Do=50
    for u in range(0,x):
        for v in range(0,y):
            duv=np.sqrt(((u-midpointx)**2)+((v-midpointy)**2))
            maskgussine[u][v]= math.exp((-(duv)**2)/(2*(Do**2)))

    #cv2.imshow("mask",maskgussine)
    maskbutter=im2double(maskgussine)
    pil_img=(dft_shift*maskbutter)/255
    pil_img=np.fft.ifftshift(pil_img)
    pil_img=np.fft.ifft2(pil_img, axes=(0,1))
    pil_img=np.abs(pil_img).clip(0,255).astype(np.uint8)
    img = cv2.resize(pil_img, (350, 340), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()
    






#ideal high pass
def ideal_highpass():
    img = cv2.imread(IMG,0)
    dft = np.fft.fft2(img, axes=(0,1))
    dft_shift = np.fft.fftshift(dft)
    # generate spectrum from magnitude image (for viewing only)
    mag = np.abs(dft_shift)
    spec = np.log(mag) / 20
    x,y=np.shape(img)
    midpointx,midpointy=x//2 ,y//2
    maskideal=np.zeros((x,y), np.uint8)
    Do=50
    for u in range(0,x):
        for v in range(0,y):
              if np.sqrt(((u-midpointx)**2)+((v-midpointy)**2))>=Do:
                  maskideal[u][v]=255

    pil_img=(dft_shift*maskideal)/255
    pil_img=np.fft.ifftshift(pil_img)
    pil_img=np.fft.ifft2(pil_img, axes=(0,1))
    pil_img=np.abs(pil_img).clip(0,255).astype(np.uint8)
    img = cv2.resize(pil_img, (350, 340), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()


 #butterworth highpass

def Butterworth_highwpass():
    def im2double(im):
        max_val = np.max(im)
        min_val = np.min(im)
        return np.round((im.astype('float') - min_val) / (max_val - min_val) * 255)
    img = cv2.imread(IMG,0)
    dft = np.fft.fft2(img, axes=(0,1))
    dft_shift = np.fft.fftshift(dft)
    # generate spectrum from magnitude image (for viewing only)
    mag = np.abs(dft_shift)
    spec = np.log(mag) / 20
    x,y=np.shape(img)
    midpointx,midpointy=x//2 ,y//2
    maskbutter=np.zeros((x,y))
    Do=50
    for u in range(0,x):
        for v in range(0,y):
                  duv=np.sqrt(((u-midpointx)**2)+((v-midpointy)**2))
                  maskbutter[u][v]=1/(1+(Do/duv)**4)

    #cv2.imshow("mask",maskbutter)
    maskbutter=im2double(maskbutter)
    pil_img=(dft_shift*maskbutter)/255
    pil_img=np.fft.ifftshift(pil_img)
    pil_img=np.fft.ifft2(pil_img, axes=(0,1))
    pil_img=np.abs(pil_img).clip(0,255).astype(np.uint8)
    img = cv2.resize(pil_img, (350, 340), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()


# Gaussien gigh pass filter
def Gaussuien_highpass():
    def im2double(im):
        max_val = np.max(im)
        min_val = np.min(im)
        return np.round((im.astype('float') - min_val) / (max_val - min_val) * 255)
    img = cv2.imread(IMG,0)
    dft = np.fft.fft2(img, axes=(0,1))
    dft_shift = np.fft.fftshift(dft)
    # generate spectrum from magnitude image (for viewing only)
    mag = np.abs(dft_shift)
    spec = np.log(mag) / 20
    x,y=np.shape(img)
    midpointx,midpointy=x//2 ,y//2
    maskgussine=np.zeros((x,y))
    Do=50
    for u in range(0,x):
        for v in range(0,y):
            duv=np.sqrt(((u-midpointx)**2)+((v-midpointy)**2))
            maskgussine[u][v]=1- math.exp((-(duv)**2)/(2*(Do**2)))

    #cv2.imshow("mask",maskgussine)
    maskbutter=im2double(maskgussine)
    pil_img=(dft_shift*maskbutter)/255
    pil_img=np.fft.ifftshift(pil_img)
    pil_img=np.fft.ifft2(pil_img, axes=(0,1))
    pil_img=np.abs(pil_img).clip(0,255).astype(np.uint8)
    img = cv2.resize(pil_img, (350, 340), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()
    






def max_filter():
    img = cv2.imread(IMG,0)
    img = Image.fromarray((img))
    img = img.filter(ImageFilter.MaxFilter())
    img = np.array(img, dtype=np.uint8)
    img = cv2.resize(img, (350, 340), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()


def min_filter():
    img = cv2.imread(IMG, 0)
    img = Image.fromarray((img))
    img = img.filter(ImageFilter.MinFilter())
    img = np.array(img, dtype=np.uint8)
    img = cv2.resize(img, (350, 340), interpolation=cv2.INTER_AREA)
    #img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()
 
    
def midpoint_filter():
    img = cv2.imread(IMG, 0)
    img = Image.fromarray((img))  
    maxf = maximum_filter(img, 3)
    minf = minimum_filter(img, 3)
    midpoint = (maxf + minf) / 2
    img = np.array(midpoint, dtype=np.uint8)
    img = cv2.resize(img, (350, 340), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()
   
    
def median_filter():
    img = cv2.imread(IMG,0)
   # img = Image.fromarray((img))
    img =cv2.medianBlur(img,5)
    img = np.array(img, dtype=np.uint8)
    img = cv2.resize(img, (350, 340), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()

def Alpha_trimmed():
    img=cv2.imread(IMG,0).astype(float)
    alpha_trimd_img= np.zeros_like(img)
    rows,cols=img.shape
    ksize=5
    padding_size = int((ksize-1)/2)
    padding_img = cv2.copyMakeBorder(img, *[padding_size]*4, cv2.BORDER_DEFAULT)
    d=3
    for r in range(rows):
        for c in range(cols):
            alpha_trimd_img[r, c] =(1/(ksize*ksize-d))*np.sum(padding_img[r:r+ksize, c:c+ksize])
    img = np.array(alpha_trimd_img, dtype=np.uint8)
    img = cv2.resize(img, (350, 340), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()


def Arithmetic_mean():
    img = cv2.imread(IMG,0)
    img = Image.fromarray((img))
    img = img.filter(ImageFilter.BLUR())
    img = np.array(img, dtype=np.uint8)
    img = cv2.resize(img, (350, 340), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()
 


def Geometric_mean():
    img=cv2.imread(IMG,0).astype(float)
    rows, cols = img.shape
    ksize = 5
    padsize = int((ksize-1)/2)
    pad_img = cv2.copyMakeBorder(img, *[padsize]*4, cv2.BORDER_DEFAULT)
    geo_mean = np.zeros_like(img)
    for r in range(rows):
        for c in range(cols):
            geo_mean[r, c] = np.prod(pad_img[r:r+ksize, c:c+ksize])**(1/(ksize**2))
    img = np.array(geo_mean, dtype=np.uint8)
    img = cv2.resize(img, (350, 340), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()


def Harmonic_mean():
    img=cv2.imread(IMG,0).astype(float)
    rows, cols = img.shape
    ksize = 5
    padsize = int((ksize-1)/2)
    pad_img = cv2.copyMakeBorder(img, *[padsize]*4, cv2.BORDER_DEFAULT)
    har_mean = np.zeros_like(img)
    for r in range(rows):
        for c in range(cols):
            har_mean[r, c] =ksize*ksize/(np.sum(1/pad_img[r:r+ksize, c:c+ksize]))
    img = np.array(har_mean, dtype=np.uint8)
    img = cv2.resize(img, (350, 340), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()
 
    
def Contra_Harmonic_mean():
    img=cv2.imread(IMG,0).astype(float)
    rows, cols = img.shape
    ksize = 5
    padsize = int((ksize-1)/2)
    pad_img = cv2.copyMakeBorder(img, *[padsize]*4, cv2.BORDER_DEFAULT)
    contra_mean = np.zeros_like(img)
    q=-1
    for r in range(rows):
        for c in range(cols):
            contra_mean[r, c] =np.sum(np.power(pad_img[r:r+ksize, c:c+ksize],q+1)) / np.sum(np.power(pad_img[r:r+ksize, c:c+ksize],q))
    img = np.array(contra_mean, dtype=np.uint8)
    img = cv2.resize(img, (350, 340), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()
    

        
def Adaptive_mean():
    img=cv2.imread(IMG,0).astype(float)  
    kernel_size = 3 
    height , width = img.shape 
    pad_size  = int((kernel_size - 1) / 2) 
    paded_img = cv2.copyMakeBorder(img , pad_size , pad_size, pad_size , pad_size , cv2.BORDER_DEFAULT)
    adMean_image = np.zeros_like(img) 
    vn = 0 
    def adMean(img_slice , vn): 
        vl  = np.var(img_slice) 
        gxy = img_slice[1 , 1]  
        var = (vn / (vl + 0.0001)) 
        ml  = np.mean(img_slice) 
        output = gxy - (var * (gxy - ml)) 
        return output 
    for row in range(height): 
        h_start = row 
        h_end   = row + kernel_size 
        for col in range(width): 
            w_start = col 
            w_end   = col + kernel_size 
            img_slice = paded_img[h_start : h_end , w_start : w_end] 
            adMean_image[row , col] = adMean(img_slice , vn)  
    img = np.array(adMean_image)
    img = cv2.resize(img, (350, 340), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()


def Global_threshold():
    img=cv2.imread(IMG,0)  
    ret,G_thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    img = np.array(G_thresh1, dtype=np.uint8)
    img = cv2.resize(img, (350, 340), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()
    


def Adaptive_mean_threshold():
    img=cv2.imread(IMG,0)  
    thresh1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 11, 2)
    img = np.array(thresh1, dtype=np.uint8)
    img = cv2.resize(img, (350, 340), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()



def Adaptive_gaussien_threshold():
    img=cv2.imread(IMG,0)  
    thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
    filename=f"alaa.jpg"
    cv2.imwrite(filename, img)
    img = np.array(thresh2, dtype=np.uint8)
    img = cv2.resize(img, (350, 340), interpolation=cv2.INTER_AREA)
    filename=f"aaaala.jpg"
    cv2.imwrite(filename, img)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()



def OTSU_threshold():
    img=cv2.imread(IMG,0)  
    ret2,Otsu_thresh2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img = np.array(Otsu_thresh2, dtype=np.uint8)
    img = cv2.resize(img, (350, 340), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()



def sobel_edge_detection():
    img=cv2.imread(IMG,0) 
    img_blur = cv2.GaussianBlur(img, (3,3), 0) 
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)
    img = np.array(sobelxy, dtype=np.uint8)
    img = cv2.resize(img, (350, 340), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()



def canny_edge():
    img=cv2.imread(IMG,0)  
    edges = cv2.Canny(img,100,200)
    img = np.array(edges, dtype=np.uint8)
    img = cv2.resize(img, (350, 340), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()   






def Watershed():
    img = cv2.imread(IMG)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]    
   # cv2.imshow('th', thresh)
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]
    img = np.array(img, dtype=np.uint8)
    img = cv2.resize(img, (350, 340), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()   





def Kmeans():
    img=cv2.imread(IMG)  
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = img.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)
    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) 
    # number of clusters (K)
    #bestlabel = return the index for each cluster 
    k = 3
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # convert back to 8 bit values
    centers = np.uint8(centers)
    # flatten the labels array
    labels = labels.flatten()
    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]    
    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(img.shape)
    # show the image
    img = np.array(segmented_image, dtype=np.uint8)
    img = cv2.resize(img, (350, 340), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()   






def connected_component_label():
    
    img = cv2.imread(IMG, 0)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    num_labels, labels = cv2.connectedComponents(img)
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue==0] = 0
    img = np.array(labeled_img, dtype=np.uint8)
    img = cv2.resize(img, (350, 340), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()   




#Mean shift algorithm
def Mean_Shift(): 
    #Loading original image
    img = cv2.imread(IMG)
    # Shape of original image    
    originShape = img.shape
    # Converting image into array of dimension [nb of pixels in originImage, 3]
    # based on r g b intensities    
    flatImg=np.reshape(img, [-1, 3])  
    # Estimate bandwidth for meanshift algorithm    
    bandwidth = estimate_bandwidth(flatImg, quantile=0.1, n_samples=100)   
    #MeanShoft take bandwidth and number of seeding samples which is selected randomly from pixels   
    ms = MeanShift(bandwidth = bandwidth, bin_seeding=True)
    # Performing meanshift on flatImg    
    ms.fit(flatImg)
    # (r,g,b) vectors corresponding to the different clusters after meanshift    
    labels=ms.labels_
    # Remaining colors after meanshift    
    cluster_centers = ms.cluster_centers_ 
    # Finding and diplaying the number of clusters    
    labels_unique = np.unique(labels)    
    #n_clusters_ = len(labels_unique) 
    # Displaying segmented image    
    segmentedImg = cluster_centers[np.reshape(labels, originShape[:2])]
    segmentedImg = np.uint8(segmentedImg )
    img = np.array(segmentedImg, dtype=np.uint8)
    img = cv2.resize(img, (350, 340), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()  

def snake():
    img = cv2.imread(IMG,0)
    s = np.linspace(0, 2*np.pi, 400)
    r = 100 + 100*np.sin(s)
    c = 220 + 100*np.cos(s)
    init = np.array([r, c]).T
    counter=1
    snake = active_contour(gaussian(img, 3, preserve_range=False), init, alpha=0.015, beta=10, gamma=0.001)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])
    filename=f"aaaaaaaaaalaa{counter+1}.jpg"
    fig.savefig(filename)   # save the figure to file
    image = Image.open(filename)
    image_tk = ImageTk.PhotoImage(image)
    image = image.resize((450, 440), Image.ANTIALIAS)
    image_tk = ImageTk.PhotoImage(image)
    canvas.create_image(-55, -52, anchor=NW, image=image_tk)
    root.mainloop()
    

def LevelSET():
        img = cv2.imread(IMG, 0) 
        img = img-np.mean(img)
        imSmooth = cv2.GaussianBlur(img,(5, 5), 0)
        img = cv2.resize(img, (390, 370), interpolation=cv2.INTER_AREA)
        image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
        canvas.create_image(0, 0, anchor=NW, image=image_tk)
        root.mainloop()




# =============================================================================
#                                   # another code for level set 
#
# import scipy.ndimage
# import scipy.signal
# import matplotlib.pyplot as plt
# from skimage import color, io
# 
# def Level_set():
#     phi = np.random.randn(20, 20) # initial value for phi
#     F = ... # some function
#     dt = 1
#     it = 100
# 
#     for i in range(100):
#         dphi = np.gradient(phi)
#         dphi_norm = np.sqrt(np.sum(dphi**2, axis=0))
# 
#         phi = phi + dt * F * dphi_norm
# 
#         # plot the zero level curve of phi
#         plt.contour(phi, 0)
#         plt.show()
#     
# 
# 
#     def grad(x):
#         return np.array(np.gradient(x))
# 
# 
#     def norm(x, axis=0):
#         return np.sqrt(np.sum(np.square(x), axis=axis))
# 
# 
#     def stopping_fun(x):
#         return 1. / (1. + norm(grad(x))**2)
# 
# 
#     img = io.imread('twoObj.bmp')
#     img = color.rgb2gray(img)
#     img = img - np.mean(img)
# 
#     # Smooth the image to reduce noise and separation between noise and edge becomes clear
#     img_smooth = scipy.ndimage.filters.gaussian_filter(img, 0)
# 
#     F = stopping_fun(img_smooth)
#     def default_phi(x):
#         # Initialize surface phi at the border (5px from the border) of the image
#         # i.e. 1 outside the curve, and -1 inside the curve
#         phi = np.ones(x.shape[:2])
#         phi[5:-5, 5:-5] = -1.
#         return phi
# 
# 
#     dt = 1.
#     for i in range(100):
#         dphi = grad(phi)
#         dphi_norm = norm(dphi)
# 
#         dphi_t = F * dphi_norm
# 
#         phi = phi + dt * dphi_t
#     def curvature(f):
#         fy, fx = grad(f)
#         norm = np.sqrt(fx2 + fy2)
#         Nx = fx / (norm + 1e-8)
#         Ny = fy / (norm + 1e-8)
#         return div(Nx, Ny)
# 
# 
#     def div(fx, fy):
#         fyy, fyx = grad(fy)
#         fxy, fxx = grad(fx)
#         return fxx + fyy
# 
# 
#     def dot(x, y, axis=0):
#         return np.sum(x * y, axis=axis)
# 
# 
#     v = 1.
#     dt = 1.
# 
#     g = stopping_fun(img_smooth, 1)
#     dg = grad(g)
# 
#     for i in range(100):
#         dphi = grad(phi)
#         dphi_norm = norm(dphi)
#         kappa = curvature(phi)
# 
#         smoothing = g * kappa * dphi_norm
#         balloon = g * dphi_norm * v
#         attachment = dot(dphi, dg)
# 
#         dphi_t = smoothing + balloon + attachment
# 
#         phi = phi + dt * dphi_t
#     img = cv2.resize(phi, (350, 340), interpolation=cv2.INTER_AREA)
#     image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
#     canvas.create_image(0, 0, anchor=NW, image=image_tk)
#     root.mainloop()
# 
#   
# =============================================================================


class Point(object):
 def __init__(self,x,y):
  self.x = x
  self.y = y

 def getX(self):
  return self.x
 def getY(self):
  return self.y

def getGrayDiff(img,currentPoint,tmpPoint):
 return abs(int(img[currentPoint.x,currentPoint.y]) - int(img[tmpPoint.x,tmpPoint.y]))

def selectConnects(p):
 if p != 0:
  connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
     Point(0, 1), Point(-1, 1), Point(-1, 0)]
 else:
  connects = [ Point(0, -1), Point(1, 0),Point(0, 1), Point(-1, 0)]
 return connects

def regionGrow(img,seeds,thresh,p = 1):
 height, weight = img.shape
 seedMark = np.zeros(img.shape)
 seedList = []
 for seed in seeds:
  seedList.append(seed)
 label = 1
 connects = selectConnects(p)
 while(len(seedList)>0):
  currentPoint = seedList.pop(0)

  seedMark[currentPoint.x,currentPoint.y] = label
  for i in range(8):
   tmpX = currentPoint.x + connects[i].x
   tmpY = currentPoint.y + connects[i].y
   if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
    continue
   grayDiff = getGrayDiff(img,currentPoint,Point(tmpX,tmpY))
   if grayDiff < thresh and seedMark[tmpX,tmpY] == 0:
    seedMark[tmpX,tmpY] = label
    seedList.append(Point(tmpX,tmpY))
 return seedMark


def rg():
    img = cv2.imread(IMG,0)
    seeds = [Point(10,10),Point(82,150),Point(20,300)]
    binaryImg = regionGrow(img,seeds,10)
    img = np.array(binaryImg, dtype=np.uint8)
    img = cv2.resize(binaryImg, (350, 340))*250
    #norm_image=cv2.normalize(img, None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()


def chainCode():    
    image = cv2.imread(IMG, 0) 
    rows , cols = image.shape
    result = np.zeros_like(image)
    for x in range(rows):
        for y in range(cols):    
            if image[x,y] >= 70:
                result[x,y] = 0
            else:
                result[x,y] = 1 
    for i, row in enumerate(result):
        for j, value in enumerate(row):
            if value == 1:
                start_point = (i, j)
    #            print(start_point, value)
                break
        else:
            continue
        break
    directions = [ 0,  1,  2,
                   7,      3,
                   6,  5,  4]
    dir2idx = dict(zip(directions, range(len(directions))))
    #print(dir2idx)
    change_j =   [-1,  0,  1, # x or columns
                  -1,      1,
                  -1,  0,  1]
    
    change_i =   [-1, -1, -1, # y or rows
                   0,      0,
                   1,  1,  1]
                                                    
    border = []
    chain = []
    curr_point = start_point
    for direction in directions:
        idx = dir2idx[direction]
        print(idx)
        new_point = (start_point[0]+change_i[idx], start_point[1]+change_j[idx])
        print(new_point)
        if result[new_point] != 0: # if is ROI
            border.append(new_point)
            chain.append(direction)
            curr_point = new_point
            break
    count = 0
    while curr_point != start_point:
        #figure direction to start search
        b_direction = (direction + 5) % 8 
        dirs_1 = range(b_direction, 8)
        dirs_2 = range(0, b_direction)
        dirs = []
        dirs.extend(dirs_1)
        dirs.extend(dirs_2)
        for direction in dirs:
            idx = dir2idx[direction]
            new_point = (curr_point[0]+change_i[idx], curr_point[1]+change_j[idx])
            if result[new_point] != 0: # if is ROI
                border.append(new_point)
                chain.append(direction)
                curr_point = new_point
                break
        if count == 1000: break
        count += 1

    plt.imshow(image, cmap='Greys')
    plt.plot([i[1] for i in border], [i[0] for i in border])
    plt.axis('off')
    plt.savefig('chain_plot.jpg',dpi=300, bbox_inches='tight')
    imgg= Image.open("chain_plot.jpg")
    img = np.array(imgg, dtype=np.uint8)
    img = cv2.resize(img, (350, 350), interpolation=cv2.INTER_AREA)
    image_tk = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, anchor=NW, image=image_tk)
    root.mainloop()
    
    

#------------------- Graphical User Interface ---------------
    
root=Tk()
root.title("Image Processing Project")
#root.iconbitmap('E:\3 Medical\GUI')
root.geometry("720x600+380+100")

# Define Image
bg=PhotoImage(file='bg6.png')
click_btn1= PhotoImage(file='google.jpg')
photoimage1 = click_btn1.subsample(3, 3)




# create a label
my_label=Label(root,image=bg)
my_label.place(x=0,y=0,relwidth=1,relheight=1)
# Label
lb=Label(root,text="NOISE FIGHTERS",bg="#001541",
         fg="#E8CEA6",bd=7,relief="raise",font=("Segoe Print",24,"bold"))
           #"ridge"  "sunken"  "flat"  "raise" "groove"
lb.pack()

# choose image
bt1=Button(root,text="Choose Image",bg="#E8CEA6",
         fg="#001541",bd=7,relief="groove",font=("Segoe Print",15,"bold"),command=choose_image)
bt1.place(x=280, y=522)

# google button
bt3=Button(root,bg="#E8CEA6",text="Google",
         fg="#001541",bd=10,relief="groove",font=("Segoe Print",13,"bold"),command=search_google)
bt3.place(x=550,y=320)

# original image
B4 = Button(root, text="Original IMG",bd=10,
            relief="groove",font=("Segoe Print",13,"bold"), fg='#001541', bg='#E8CEA6', command=Remove)
B4.place(x=550, y=400)

# Exit
bt2=Button(root,text="Exit",bg="#E8CEA6",
         fg="#001541",bd=7,relief="groove",font=("Segoe Print",13,"bold"),command=Exit)
bt2.place(x=550,y=480)

# # Label
# lb=Label(root,text="Welcome to Image Processing GUI",bg="white",
#          fg="orangered",bd=7,relief="flat",font=("Comic Sans MS",20,"bold"))
#            #"ridge"  "sunken"  "flat"  "raise" "groove"
# lb.pack()


# bt1=Button(root,text="Choose Image",bg="white",
#          fg="navy blue",bd=10,relief="raised",font=("Comic Sans MS",15,"bold"),command=choose_image)
# bt1.place(x=150,y=52)

# bt2=Button(root,text="Exit",bg="white",
#          fg="red",bd=7,relief="raised",font=("Comic Sans MS",20,"bold"),command=Exit)
# bt2.place(x=450,y=470)

# bt3=Button(root,bg="white",
#          fg="red",bd=10,relief="raised",font=("Comic Sans MS",15,"bold"),image=photoimage1,command=search_google)
# bt3.place(x=450,y=52)

# B4 = Button(root, text="Re Noise",bd=10,
#             relief="raised",font=("Comic Sans MS",17,"bold"), fg='red', bg='white', command=Remove)
# B4.place(x=150, y=470)


#bt5=Button(root,text="Save As",command=save_file).pack()

# Menu List
calc=Frame(root)
calc.pack()

menubar=Menu(calc)
smoothingFRQ_Filters=Menu(menubar,tearoff=0)
SHarpiningFRQ_filters=Menu(menubar,tearoff=0)
Mean_Filters=Menu(menubar,tearoff=0)
Statistic_Filters=Menu(menubar,tearoff=0)
Adaptive_Filters=Menu(menubar,tearoff=0)
Segmentation=Menu(menubar,tearoff=0)



menubar.add_cascade(label="Smoothing Frq Filters",menu=smoothingFRQ_Filters)
menubar.add_cascade(label="Sharpening Frq Filters",menu=SHarpiningFRQ_filters)
menubar.add_cascade(label="Mean Filters",menu=Mean_Filters)
menubar.add_cascade(label="Statistic Filters",menu=Statistic_Filters)
menubar.add_cascade(label="Adaptive Filters",menu=Adaptive_Filters)
menubar.add_cascade(label="Segmentation",menu=Segmentation)


smoothingFRQ_Filters.add_command(label="Ideal lowpass Filter",command=ideal_lowpass)
smoothingFRQ_Filters.add_command(label="Gaussian lowpass Filter",command=Gaussuien_lowpass) 
smoothingFRQ_Filters.add_command(label="Butterworth lowpass Filter",command=Butterworth_lowpass)


SHarpiningFRQ_filters.add_command(label="Ideal highpass Filter",command=ideal_highpass)
SHarpiningFRQ_filters.add_command(label="Gaussian highpass Filter",command=Gaussuien_highpass)
SHarpiningFRQ_filters.add_command(label="Butterworth highpass Filter",command=Butterworth_highwpass)


Mean_Filters.add_command(label="Harmonic Mean Filter",command=Harmonic_mean)
Mean_Filters.add_command(label="Geometric Mean Filter",command=Geometric_mean)
Mean_Filters.add_command(label="Arithmetic Mean Filter",command=Arithmetic_mean)
Mean_Filters.add_command(label="Controharmonic Mean Filter",command=Contra_Harmonic_mean)
#Mean_Filters.add_separator()


Statistic_Filters.add_command(label="Max Filter",command=max_filter)
Statistic_Filters.add_command(label="Min Filter",command=min_filter)
Statistic_Filters.add_command(label="Median Filter",command=median_filter)
Statistic_Filters.add_command(label="Midpoint Filter",command=midpoint_filter)
Statistic_Filters.add_command(label="Alpha-Trimmed Mean Filter",command=Alpha_trimmed)


Adaptive_Filters.add_command(label="Adaptive mean filter",command=Adaptive_mean)


Segmentation.add_command(label="OTSU Threshold",command=OTSU_threshold)
Segmentation.add_command(label="Global Threshold",command=Global_threshold)
Segmentation.add_command(label="Adaptive Mean Threshold",command=Adaptive_mean_threshold)
Segmentation.add_command(label="Adaptive Gaussien Threshold",command=Adaptive_gaussien_threshold)
Segmentation.add_separator()

Segmentation.add_command(label="Sobel Edge Detection",command=sobel_edge_detection)
Segmentation.add_command(label="Canny Edge Detection",command=canny_edge)
Segmentation.add_separator()


Segmentation.add_command(label="Level SET",command=LevelSET)
Segmentation.add_command(label="Mean Shift",command=Mean_Shift)
Segmentation.add_command(label="Region Growing",command=rg)
Segmentation.add_command(label="Snake Algorithm",command=snake)
Segmentation.add_command(label="Kmeans Algorithm",command=Kmeans)
Segmentation.add_command(label="Whatershed Algorithm",command=Watershed)
Segmentation.add_command(label="Connected Component Label",command=connected_component_label)
Segmentation.add_command(label="Chain Code",command=chainCode)


root.configure(menu=menubar)
  
root.mainloop();
