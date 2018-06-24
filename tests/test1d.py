import sys
import time
import argparse
import traceback
import pywt
import numpy
import scipy.misc
from skimage.transform import resize
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.tools import DeviceMemoryPool

from pycudaDWT import PycudaWaveletTransform

# root mean square error
def rmse(a, b):
    return numpy.sqrt(((a-b)**2).mean())

def test1d(wavelet='haar', use_float32=False, depth=1, num_rows=512, row_size=512,
           iterations=20, gpu_input=False, gpu_output=False, gpu_mempool=False):
    try:
        dtype = numpy.float64
        if use_float32:
            dtype = numpy.float32
        img = (numpy.array(scipy.misc.ascent(), dtype=dtype)-128.)/128.
        resized_img = resize(img, (num_rows, row_size), mode='constant')

        if gpu_input:
            cont_input_array = numpy.ascontiguousarray(resized_img, dtype=dtype)
            img_array_gpu = gpuarray.to_gpu(cont_input_array)
        else:
            img_array_gpu = resized_img

        if gpu_mempool:
            dev_mem_pool = DeviceMemoryPool()
            gpu_alloc = dev_mem_pool.allocate
        else:
            gpu_alloc = cuda.mem_alloc

        pwt = PycudaWaveletTransform(wavelet=wavelet, use_float32=use_float32)

        # Forward Transform
        print('---------FORWARD DWT---------')
        t = time.time()
        for _ in range(iterations):
            dec_cpu = pywt.wavedec(resized_img, wavelet=wavelet, mode='periodization', level=depth)
        t = time.time()-t
        print('PyWavelets:\t\t\t\t{:.3f} ms'.format((t*1000.)/iterations))

        t = time.time()
        for _ in range(iterations):
            dec_gpu = pwt.dwt1d(img_array_gpu, depth=depth, gpu_output=gpu_output, gpu_allocator=gpu_alloc)
        t = time.time()-t
        print('PycudaWaveletTransform:\t{:.3f} ms'.format((t*1000.)/iterations))

        for i, (d1, d2) in enumerate(zip(dec_gpu, dec_cpu)):
            if i == 0:
                result1 = d1.get() if gpu_output else d1
                result2 = d2
            else:
                result1 = numpy.concatenate((result1, d1.get() if gpu_output else d1), axis=1)
                result2 = numpy.concatenate((result2, d2), axis=1)
        print('RMSE: {} \n'.format(rmse(result1, result2)))

        dec_cpu_g = []
        if gpu_input:
            for d in dec_cpu:
                cont_array = numpy.ascontiguousarray(d, dtype=dtype)
                dec_cpu_g.append(gpuarray.to_gpu(cont_array))
        else:
            dec_cpu_g = dec_cpu

        # Inverse Transform
        print('---------INVERSE DWT---------')
        t = time.time()
        for _ in range(iterations):
            rec_cpu = pywt.waverec(dec_cpu, wavelet=wavelet, mode='periodization')
        t = time.time()-t
        print('PyWavelets:\t\t\t\t{:.3f} ms'.format((t*1000.)/iterations))

        t = time.time()
        for _ in range(iterations):
            rec_gpu = pwt.idwt1d(dec_cpu_g, gpu_output=gpu_output, gpu_allocator=gpu_alloc)
        t = time.time()-t
        print('PycudaWaveletTransform:\t{:.3f} ms'.format((t*1000.)/iterations))

        print('RMSE: {} '.format(rmse(rec_gpu.get() if gpu_output else rec_gpu, rec_cpu)))

        if gpu_mempool:
            dev_mem_pool.stop_holding()

    except Exception as e:
        tb = traceback.format_exc()
        print("%s",tb)


def main(argv):
    parser = argparse.ArgumentParser(description='1D DWT test of PycudaDWT')
    parser.add_argument('--dev_id', '-di', dest='dev_id', type=int, default=0, help='CUDA device id to use')
    parser.add_argument('--wavelet', '-w', dest='wavelet', type=str, default='haar', help='Wavelet type')
    parser.add_argument('--use_float32', dest='use_float32', action='store_true', help='Use 32-bit floating point instead of 64-bit floating point')
    parser.add_argument('--depth', '-d', dest='depth', type=int, default=1, help='Depth of transformation')
    parser.add_argument('--num_rows', '-nr', dest='num_rows', type=int, default=512, help='Number of rows')
    parser.add_argument('--row_size', '-rs', dest='row_size', type=int, default=512, help='Size of rows')
    parser.add_argument('--iterations', '-i', dest='iterations', type=int, default=20, help='Number of iterations for time profiling')
    parser.add_argument('--gpu_input', dest='gpu_input', action='store_true', help='Use GPUArray for input arrays')
    parser.add_argument('--gpu_output', dest='gpu_output', action='store_true', help='Return results in GPUArray objects')
    parser.add_argument('--gpu_mempool', dest='gpu_mempool', action='store_true', help='Use DeviceMemoryPool for device memory allocation')
    parser.set_defaults(use_float32=False, gpu_input=False, gpu_output=False, gpu_mempool=False)
    args = parser.parse_args(argv)

    # Init CUDA
    cuda.init()
    cuda_device  = cuda.Device(args.dev_id)
    cuda_context = cuda_device.make_context()

    test1d(wavelet=args.wavelet, use_float32=args.use_float32, depth=args.depth,
           num_rows=args.num_rows, row_size=args.row_size, iterations=args.iterations,
           gpu_input=args.gpu_input, gpu_output=args.gpu_output, gpu_mempool=args.gpu_mempool)

    # Detach CUDA before leaving
    cuda_context.detach()

if __name__ == "__main__":
    main(sys.argv[1:])