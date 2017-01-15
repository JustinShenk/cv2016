
Osnabrück University - Computer Vision (Winter Term 2016/17) - Prof. Dr.-Ing. G. Heidemann, Ulf Krumnack

# Exercise Sheet 07: Fourier Transform

## Introduction

This week's sheet should be solved and handed in before the end of **Sunday, January 15, 2017**. If you need help (and Google and other resources were not enough), feel free to contact your groups' designated tutor or whomever of us you run into first. Please upload your results to your group's Stud.IP folder.

## Assignment 0: Math recap (Expectation and Variance) [0 Points]

This exercise is supposed to be very easy, does not give any points, and is voluntary. There will be a similar exercise on every sheet. It is intended to revise some basic mathematical notions that are assumed throughout this class and to allow you to check if you are comfortable with them. Usually you should have no problem to answer these questions offhand, but if you feel unsure, this is a good time to look them up again. You are always welcome to discuss questions with the tutors or in the practice session. Also, if you have a (math) topic you would like to recap, please let us know.

**a)** What is the relation between mean and expectated value? How to compute it?
FIXME: change cell type to 'Markdown' and put your answer here!
**b)** What is the variance? What does it express? Why is there a square in the formula?
FIXME: change cell type to 'Markdown' and put your answer here!
**c)** Can you compute mean and variance of a given 1-dimensional dataset (e.g., $D=\{9,10,11,7,13\}$). Can you do the same for a 3-dimensional dataset (e.g., D=\{(1,10,9), (1,10,10), (10,10,11), (19,10,7), (19,10,13)\})?
FIXME: change cell type to 'Markdown' and put your answer here!
## Exercise 1 (Understanding Fourier Transform – 6p)

This exercise aims at getting some intuition of finite, 2d-Fourier transform.

**a)** Transform the image `dolly.png` into the frequency space (you may use the function
`numpy.fft.fft2`). The result will be a complex matrix. Plot histograms for the amplitude and phase
values.


```python
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from numpy.fft import fft2,fftshift
img = misc.imread('dolly.png', mode = 'F')

# Get frequency space of image.
freqSpace = fft2(img)
freqSpace = fftshift(freqSpace)

# Convert to amplitude.
amp = np.log(np.abs(freqSpace))

# # Get amplitude.
# histAmp, bins = np.histogram(amp.flatten(),normed=True)
# center = (bins[:-1] + bins[1:]) / 2

# Get phase.
phase = np.angle(freqSpace)

# Display.
plt.gray()
plt.subplot(2,3,1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')
plt.subplot(2,3,2)
plt.hist(amp.flatten(),1000,normed=1)
plt.title('Amplitude')
plt.subplot(2,3,3)
plt.hist(phase.flatten(),1000,normed=1)
plt.title('Phase')
plt.rc('xtick', labelsize=8)          # fontsize of the tick labels
plt.rc('ytick', labelsize=8)          # fontsize of the tick labels
plt.show()
```


![png](sheet07_files/sheet07_12_0.png)


**b)** Display the amplitude and phase in separate images. You may again take the logarithm of
the amplitude to enhance the contrast. You may also center the base frequency (see Matlab
function fftshift).


```python
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from numpy.fft import fft2,fftshift
img = misc.imread('dolly.png', mode = 'F')

# Get frequency space of image.
freqSpace = fft2(img)
freqSpace = fftshift(freqSpace)
amp = np.abs(freqSpace)

# Normalize.
# amp /= np.max(np.abs(amp),axis=0)
amp = np.log(abs(freqSpace))
# amp = np.divide(amp,np.amax(amp))
phase = np.angle(freqSpace)
        
fig = plt.figure()
plt.gray()
plt.subplot(2,3,1)
plt.imshow(img)
plt.title('Original')
plt.subplot(2,3,2)
plt.imshow(amp)
plt.title('Amplitude')
plt.subplot(2,3,3)
plt.imshow(phase)
plt.title('Phase')

# plt.subplot(2,3,4)
# amp = np.exp(amp)
# sortedAmp = amp.flatten()
# print (sortedAmp)
# # Take only 100 largest amplitudes.
# sortedAmp = sorted(sortedAmp, reverse=True)
# amp[amp<sortedAmp[100]] = 0
# plt.plot(amp)
fig.autofmt_xdate()
# plt.imshow(sortedAmp)
plt.rc('xtick', labelsize=8)          # fontsize of the tick labels
plt.rc('ytick', labelsize=8)          # fontsize of the tick labels
plt.show()
```


![png](sheet07_files/sheet07_14_0.png)


**c)** Transform the image back from the frequency space to the image space (again using `fft2`).
What do you observe? Explain and repair the result.


```python
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from numpy.fft import fftshift

img = misc.imread('dolly.png', mode = 'F')
freqSpace_centered = fftshift(img)
transformed = np.fft.fft2(freqSpace_centered)
freqstuff = np.abs(transformed)

#image rotated 90 degrees -> flip upside down and left right
transformed1 = transformed[-1:0:-1,:]
transformed2 = transformed1[:,-1:0:-1]
transformed3 = np.abs(transformed2)

#then just plot it in grey
plt.gray()
plt.subplot(1,2,1)
plt.title('Centered')
plt.imshow(np.log(freqstuff))

plt.subplot(1,2,2)
plt.title('transformed')
plt.imshow(np.log(transformed3))
plt.rc('xtick', labelsize=8)          # fontsize of the tick labels
plt.rc('ytick', labelsize=8)          # fontsize of the tick labels
plt.show()
```


![png](sheet07_files/sheet07_16_0.png)


**d)** Now restore the image, but only keep the amplitude and vary the phase. Try fixed phase
values (0, $\pi/2$,. . . ), a random phase matrix, or a noisy version of the original phase values.


```python
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

img = misc.imread('dolly.png', mode = 'F')
plt.gray()

# Keep the amplitude.
amp = abs(freqSpace_centered)
amp /= np.max(np.abs(amp),axis=0)

# Phase 0
amp_zeros = np.zeros((freqSpace_centered.shape))
fourier1 = np.multiply(amp, np.cos(amp_zeros)) + 1j * np.sin(amp_zeros);
img1 = np.fft.ifft2(fourier1)
plt.subplot(2,2,1)
img2=np.abs(img1)
plt.imshow(img2)
plt.title('Phase = 0')

# Phase pi/2
pi_half = np.zeros((freqSpace_centered.shape))
pi_half[:,:] = np.pi/2
fourier2 = np.multiply(amp, np.cos(pi_half)) + 1j * np.sin(pi_half);
img3 = np.fft.ifft2(fourier2)
plt.subplot(2,2,2)
img4=np.abs(img3)
plt.imshow(img4)
plt.title('Phase = pi/2')

# Phase pi
pi = np.zeros((freqSpace_centered.shape))
pi[:,:] = np.pi/2
fourier3 = np.multiply(amp, np.cos(pi)) + 1j * np.sin(pi);
img5 = np.fft.ifft2(fourier3)
plt.subplot(2,2,3)
img6=np.abs(img5)
plt.imshow(img6)
plt.title('Phase = pi')

plt.subplot(2,2,4)
plt.imshow(amp, plt.cm.gray)
plt.title('Amplitude')
plt.tight_layout()
plt.rc('xtick', labelsize=8)          # fontsize of the tick labels
plt.rc('ytick', labelsize=8)          # fontsize of the tick labels
plt.show()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-9-cd6c9d29e583> in <module>()
          7 
          8 # Keep the amplitude.
    ----> 9 amp = abs(freqSpace_centered)
         10 amp /= np.max(np.abs(amp),axis=0)
         11 


    NameError: name 'freqSpace_centered' is not defined


**e)** Do the same, but now keep the phase while varying the amplitude values. Again try constant,
amplitude, randomly distributed amplitudes and noisy versions of the the original values.



```python
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

img = misc.imread('dolly.png', mode = 'F')

plt.gray()
plt.subplot(2,2,1)
plt.imshow(img,plt.cm.gray)
plt.title('Original image')
 
#constant amplitude 
amp1 = np.zeros(np.shape(freqSpace_centered))
amp1[:,:] = np.exp(10)
#amp1[:,:] = np.pi
fourier4 = np.multiply(amp, np.cos(phase)) +1j * np.sin(phase)
img7 = np.fft.ifft2(fourier4)
img8 = np.abs(img7)
plt.subplot(2,2,2)
plt.imshow(img8)
plt.title('constant amplitude: exp(10)')
 
#get min and max for random distribution
amp = np.abs(freqSpace_centered)
amp = np.log(amp+1)
min_amp = np.min(np.min(amp))
max_amp = np.max(np.max(amp))
#randomly chosen amplitude within distribution bounds
amp = min_amp + np.multiply(max_amp-min_amp,np.random.rand(600,800)) #maybe size not shape
amp = np.exp(amp)
fourier = np.multiply(amp, np.cos(phase)) + 1j * np.sin(phase)
img = np.fft.ifft2(fourier)
img=np.abs(img)
plt.subplot(2,2,3)
plt.imshow(img)
plt.title('random amplitude')
                          
#% noisy version of original amplitude
#amp = abs(f_centered);
#amp = log(amp+1);
#min_amp = min(min(amp));
#max_amp = max(max(amp));
 
#amp = amp + normrnd(min_amp+(max_amp-min_amp)/2, (max_amp-min_amp)/20, size(f_centered));
#amp = exp(amp);
#% put together 
#fourier = amp .* cos(phase) + 1i * sin(phase);
#img = ifft2(fourier);
#subplot(2,2,4)
#imshow(mat2gray(abs(img)))
#title('noisy original amplitude')
plt.tight_layout()
plt.rc('xtick', labelsize=8)          # fontsize of the tick labels
plt.rc('ytick', labelsize=8)          # fontsize of the tick labels
plt.show()
```


![png](sheet07_files/sheet07_20_0.png)


*Hint:* Python and numpy can deal with complex numbers: `np.real()` and `np.imag()` provide the real and imaginary parts. `np.abs()` and `np.angle()` provide amplitude and phase. `np.conj()` gives the complex conjugate.

## Exercise 2 (Implementing Fourier Transform – 8p)

**a)** 
Explain in your own words the idea of Fourier transform. What is the frequency space? What does a point in that space represent?
Fourier transform reduces a matrix to composite sine and cosine functions. A frequency space shows the distribution of high and low frequency wave components.
**b)** First implement a one-dimensional discrete version of Fourier transform, i.e. use the formula
$$ c_n = \sum_{x=0}^{L-1} f(x)\cdot e^{\tfrac{2\pi i\cdot n}{L}\cdot x} \qquad \text{for $n=0,\ldots,L-1$}$$
for complex valued coefficients.

Plot the graph and the results of your Fourier transform, using the Matplotlib function `plot()`, for different functions. Compare your results with the output of the function `numpy.fft.fft`.


```python
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

def fourier1d(func):
    """
    Perform a discrete 1D Fourier transform.
    
    """
    # Initialize empty fourtier transform output.
    ft = np.zeros(func.shape, dtype = np.complex)
    x = np.asarray(func, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N) 
    ft = np.dot(M,x)
    return ft

# number of points
L = np.arange(100)

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

func = np.sin(2*np.pi*L/len(L))
# func = np.zeros(L.shape); func[40:60] = 1;
# func = gaussian(L, 0, 10)

ft = fourier1d(func)
ft2 = np.fft.fft(func)
plt.figure(figsize=(12,8))

plt.gray()
plt.subplot(2,3,1)
plt.plot(L,func)
plt.title('Function')
plt.subplot(2,3,2)
plt.plot(L,np.abs(ft))
plt.title('FT (Amplitude)')
plt.subplot(2,3,3)
plt.plot(L,np.angle(ft))
plt.title('FT (Frequency)')
plt.subplot(2,3,4)
plt.plot(L,np.abs(ft2))
plt.title('Amplitude - Numpy `fft` function')
plt.subplot(2,3,4)
plt.plot(L,np.angle(ft2))
plt.title('Phase Numpy `fft` function')
plt.show()
```


![png](sheet07_files/sheet07_26_0.png)


**c)** Now implement a 2-dimensional version of Fourier transform for images, using the formula from the lecture. Compare your result with the output of `fft2`.


```python
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

def fourier2d(img):
    """
    Perform discrete 2D Fourier transform of a given image.
    """

    # Initialize empty fourtier transform output.
    ft = np.zeros(img.shape, dtype = np.complex)
    x = np.asarray(img, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N) 
    ft = np.dot(abs(M),x)
    return ft

img = misc.imread('dolly.png', mode = 'F')
transformed = fourier2d(img)
plt.gray()
plt.imshow(transformed)
# plt.figure()
# plt.imshow(np.fft.fft2(img)) # Gives `Image data cannot convert to float` error.
plt.show()
```


![png](sheet07_files/sheet07_28_0.png)


## Exercise 3 (Applying Fourier Transform – 6p)

1. Read the image `text_deg.jpg`, display it and apply Fourier transform. The resulting amplitude should show the angle of the text.

2. Try to automatically get the rotation angle from the Fourier space. There are different ways to achieve this.
   Hints:
   * You may apply an erosion operation to strengthen the text sections and thereby get
     better (i.e. less noisy) amplitude values.
   * You may threshold the amplitudes, to only keep “relevant” values. You can then compute the angle of the largest relevant value.
   * Alternatively, you may apply methods you know from other lectures to get the main component and compute its angle.

3. Rotate the image back to its originally intended orientation (`scipy.misc.imrotate`).


```python
import numpy as np
from scipy import misc
from skimage.morphology import erosion
import matplotlib.pyplot as plt
from skimage.transform import hough_line,hough_line_peaks
text = misc.imread('text_deg.jpg', mode = 'F')
fig = plt.figure()
plt.gray() 
text = erosion(text)
plt.subplot(2,2,1)
plt.imshow(text2)
plt.title('Eroded')
plt.subplot(2,2,2)
fourier = np.fft.fft2(text)
fourier = np.fft.fftshift(fourier)
amp = np.log(abs(fourier))
plt.imshow(amp)
plt.title('Logarithmic Amplitude')
plt.subplot(2,2,3)
amp = np.exp(amp)
sortedAmp = amp.flatten()
sortedAmp = sorted(sortedAmp,reverse=True)
amp1 = amp.copy()
amp1[amp<sortedAmp[100]] = 0
plt.imshow(amp1,vmax=.0001)
plt.title('100 Largest Amplitudes')

# Use strait-line Hough transform to get the angle.
ax = plt.gca()
rows, cols = amp1.shape
h, theta, d = hough_line(amp1)
for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y1 = (dist - cols * np.cos(angle)) / np.sin(angle)
    ax.plot((0, cols), (y0, y1), '-r')
    print (angle, 'radians, ','{} degrees'.format(angle*180/np.pi))
ax.axis((0, cols, rows, 0))
ax.set_title('Detected line\nAngle = {} degrees'.format(angle*180/np.pi))
ax.set_axis_off()
plt.tight_layout()
plt.show()
```

    1.02672162142 radians,  58.82681564245809 degrees



![png](sheet07_files/sheet07_30_1.png)



```python

```
