import argparse
import numpy as np
from enum import Enum
from skimage import io, img_as_float
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

class svd(Enum):
    custom = "custom"
    library = "library"

    def __str__(self):
        return self.name

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',  type=str, help="plik z oryginalnym obrazkiem")
    parser.add_argument('-out',required=False, type=str,  default=None, help="nazwa pliku wyjściowego, do którego zapisany ma być skompresowany obrazek")
    parser.add_argument('-svd', required=False,  type=svd, default=svd.library, help="implementacja SVD do użycia. Możliwe wartości:’custom’(domyślna),’library’")
    parser.add_argument('-k', required=False, type=int, default=-1, help="liczba wartości osobliwych użyta do kompresji (domyślnie wszystkie, czyli brak kom-presji (-1))")

    args = parser.parse_args()
    return args

def lib_svd(img,k,out):
    
    image_reshaped = img.reshape((img.shape[0], img.shape[1] * 3))

    u, s, vh = np.linalg.svd(image_reshaped, full_matrices=True)

    compressed = np.dot(u[:,:k],np.dot(np.diag(s[:k]),vh[:k,:]))

    compressed = compressed.reshape(img.shape)

    variation = np.diag(s[:k]).sum()/np.diag(s).sum()

    fig, axes = plt.subplots(1,2,figsize=(15,10))    
    ax = axes.ravel()

    ax[0].imshow(compressed)
    ax[0].set_title("Compressed:"+ str(1-variation)+" k="+str(k))
    ax[1].imshow(img)
    ax[1].set_title("Orginal")

    fig.suptitle('np.linalg.svd', fontsize=16)
    if out:
        plt.savefig(out)
    else:
        plt.show()

def padding(s, shape):
    temp_s = s.copy()
    if shape[0] > s.shape[0]:
        temp_s = np.pad(temp_s, ((0, shape[0]-s.shape[0]),(0,0)), 'constant', constant_values=0)
    if shape[1] > s.shape[1]:
        temp_s = np.pad(temp_s, ((0,0),(0, shape[1]-s.shape[1])), 'constant', constant_values=0)
    return temp_s[:shape[0],:shape[1]]

def custom_svd(img,k,out):

    image_reshaped = img.reshape((img.shape[0], img.shape[1] * 3))

    # X^TX
    imgTimg = np.dot(image_reshaped.T, image_reshaped)

    eigval,eigvec = np.linalg.eigh(imgTimg)

    order_eigval = np.argsort(eigval)[::-1]
    sorted_values = eigval[order_eigval]

    #V
    vh =  eigvec[:, order_eigval]
    
    # Sigma
    s = np.diag(np.sqrt(sorted_values))    
    s = padding(s,image_reshaped.shape)

    # Sigma + 
    sigma_p = np.where(s.T==0,0,1/s.T) 

    # U 
    u = np.dot(image_reshaped, np.dot(vh, sigma_p))

    compressed = np.dot(u[:, :k], np.dot(s[:k, :k], vh.T[:k, :]))

    compressed = compressed.reshape(img.shape)

    variation = np.diag(s[:k]).sum()/np.diag(s).sum()

    fig, axes = plt.subplots(1,2,figsize=(15,10))    
    ax = axes.ravel()

    ax[0].imshow(compressed)
    ax[0].set_title("Compressed:"+ str(1-variation)+" k="+str(k))
    ax[1].imshow(img)
    ax[1].set_title("Orginal")
    fig.suptitle('Custom svd', fontsize=16)
    
    if out:
        plt.savefig(out)
    else:
        plt.show()

if __name__ == "__main__":
    parsed_args = parseArguments()

    img = img_as_float(io.imread(parsed_args.f))

    k=img.shape[0]
    if parsed_args.k != -1:
        k = parsed_args.k

    out = parsed_args.out

    if parsed_args.svd == svd.library:
        lib_svd(img,k,out)
    elif parsed_args.svd == svd.custom:
        custom_svd(img,k,out)