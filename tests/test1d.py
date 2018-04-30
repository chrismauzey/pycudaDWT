import sys
import time
import argparse
import traceback
import pywt
import numpy
import scipy.misc
from skimage.transform import resize
import pycuda.driver as cuda

from pycudaDWT import PycudaWaveletTransform

# root mean square error
def rmse(a,b):
    return numpy.sqrt(((a-b)**2).mean())

def test1d(wavelet='haar', use_float32=False, depth=1, num_rows=512, row_size=512, iterations=20):
    try:
        dtype = numpy.float64
        if use_float32:
            dtype = numpy.float32
        img = (numpy.array(scipy.misc.ascent(), dtype=dtype)-128.)/128.
        resized_img = resize(img, (num_rows, row_size), mode='constant')

        pwt = PycudaWaveletTransform(wavelet=wavelet, use_float32=use_float32)

        # Forward Transform
        print('---------FORWARD DWT---------')
        t = time.time()
        for _ in range(iterations):
            dec_cpu = pywt.wavedec(resized_img,wavelet=wavelet,mode='periodization',level=depth)
        t = time.time()-t
        print('PyWavelets:\t\t\t\t{:.3f} ms'.format((t*1000.)/iterations))

        t = time.time()
        for _ in range(iterations):
            dec_gpu = pwt.dwt1d(resized_img,depth=depth)
        t = time.time()-t
        print('PycudaWaveletTransform:\t{:.3f} ms'.format((t*1000.)/iterations))

        for i, (d1, d2) in enumerate(zip(dec_gpu, dec_cpu)):
            if i == 0:
                result1 = d1
                result2 = d2
            else:
                result1 = numpy.concatenate((result1,d1), axis=1)
                result2 = numpy.concatenate((result2,d2), axis=1)
        print('RMSE: {} \n'.format(rmse(result1, result2)))

        # Inverse Transform
        print('---------INVERSE DWT---------')
        t = time.time()
        for _ in range(iterations):
            rec_cpu = pywt.waverec(dec_cpu,wavelet=wavelet,mode='periodization')
        t = time.time()-t
        print('PyWavelets:\t\t\t\t{:.3f} ms'.format((t*1000.)/iterations))

        t = time.time()
        for _ in range(iterations):
            rec_gpu = pwt.idwt1d(dec_cpu)
        t = time.time()-t
        print('PycudaWaveletTransform:\t{:.3f} ms'.format((t*1000.)/iterations))

        print('RMSE: {} '.format(rmse(rec_gpu,rec_cpu)))

    except Exception as e:
        tb = traceback.format_exc()
        print("%s",tb)


def main(argv):
    parser = argparse.ArgumentParser(description='1D DWT test of PycudaDWT')
    parser.add_argument('--dev_id', '-di', dest='dev_id', type=int, default=0, help='CUDA device id to use')
    parser.add_argument('--wavelet', '-w', dest='wavelet', type=str, default='haar', help='Wavelet type')
    parser.add_argument('--use_float32', dest='use_float32', action='store_true')
    parser.add_argument('--depth', '-d', dest='depth', type=int, default=1, help='Depth of transformation')
    parser.add_argument('--num_rows', '-nr', dest='num_rows', type=int, default=512, help='Number of rows')
    parser.add_argument('--row_size', '-rs', dest='row_size', type=int, default=512, help='Size of rows')
    parser.add_argument('--iterations', '-i', dest='iterations', type=int, default=20, help='Number of iterations for time profiling')
    parser.set_defaults(use_float32=False)
    args = parser.parse_args(argv)

    # Init CUDA
    cuda.init()
    cuda_device  = cuda.Device(args.dev_id)
    cuda_context = cuda_device.make_context()

    test1d(wavelet=args.wavelet, use_float32=args.use_float32, depth=args.depth, num_rows=args.num_rows, row_size=args.row_size, iterations=args.iterations)

    # Detach CUDA before leaving
    cuda_context.detach()

if __name__ == "__main__":
    main(sys.argv[1:])