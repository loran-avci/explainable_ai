import numpy as np
import cv2

def cropp_img(X, method = "global"):
    """
    crops an MRI image into size of visible brain area
    returns: list uf cropped images
    """
    cropped = []
    thl =[]
    for i in range(X[:,:,:,:].shape[0]):
        im = X[i,:,:,:]
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.uint8)
        th = np.quantile(im, q = 0.70)
        thl.append(th)
        if method == "global":
            _, thresh = cv2.threshold(im, th, 255, 0)
        elif method == "otsu":
            blur = cv2.GaussianBlur(im,(49,49),0)
            _,thresh = cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        else:
            print('invalid method')
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c = np.vstack(contours).reshape(-1,2)
        xmin = np.min(c[:,0])
        ymin = np.min(c[:,1])
        xmax = np.max(c[:,0])
        ymax = np.max(c[:,1])
        img = im[ymin:ymax, xmin:xmax]
        #img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (192,192), interpolation = cv2.INTER_CUBIC)
        img.astype(np.float64)
        cropped.append(img)
    return(cropped,thl)