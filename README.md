# pycudaDWT

pycudaDWT is a Discrete Wavelet Transform (DWT) package for Python that uses PyCUDA to run transforms on NVIDIA GPUs.

It currently has support for batched 1D forward and inverse DWTs with future plans for 2D and 3D transforms.

# Required Python Modules
+ Numpy
+ PyCUDA
+ PyWavelets

# Example
```python
import numpy
import scipy.misc
import pycuda.driver as cuda
from pycudaDWT import PycudaWaveletTransform

# Init CUDA device and create device context
cuda.init()
cuda_device  = cuda.Device(0)
cuda_context = cuda_device.make_context()
    
# create Daubechies 3 wavelet transform
wt = PycudaWaveletTransform(wavelet='db3')

# get 512 x 512 image as 64-bit float array
img = numpy.array(scipy.misc.ascent(), dtype=numpy.float64)

# decimate image
coeff = wt.dwt2d(img,depth=4)

# reconstruct image
recon = wt.idwt2d(coeff)

# get root mean square error of the original image vs the reconstructed image
print(numpy.sqrt(((img-recon)**2).mean()))

# detach device context before leaving
cuda_context.detach()
```