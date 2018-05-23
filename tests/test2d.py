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

def test2d(wavelet='haar', use_float32=False, depth=1, num_slices=1, row_size=512, col_size=512, iterations=20):
    try:
        dtype = numpy.float64
        if use_float32:
            dtype = numpy.float32

        # Prepare Image Array
        img = (numpy.array(scipy.misc.ascent(), dtype=dtype)-128.)/128.
        resized_img = resize(img, (col_size, row_size), mode='constant')

        img_array = numpy.empty([num_slices, col_size, row_size],dtype=dtype)
        for s in range(num_slices):
            img_array[s,:,:] = resized_img[:,:]

        pwt = PycudaWaveletTransform(wavelet=wavelet, use_float32=use_float32)

        # Forward Transform
        print('---------FORWARD 2D DWT---------')
        t = time.time()
        for _ in range(iterations):
            dec_cpu = [pywt.wavedec2(img_array[s],wavelet=wavelet,mode='periodization',level=depth) for s in range(num_slices)]
        t = time.time()-t
        print('PyWavelets:\t\t\t\t{:.3f} ms'.format((t*1000.)/iterations))

        t = time.time()
        for _ in range(iterations):
            dec_gpu = pwt.dwt2d(img_array,depth=depth)
        t = time.time()-t
        print('PycudaWaveletTransform:\t{:.3f} ms'.format((t*1000.)/iterations))

        dec_cpu_g = []
        for ig, vg in enumerate(dec_gpu):
            if ig == 0:
                a = numpy.empty_like(vg)
                for ic, vc in enumerate(dec_cpu):
                    a[ic,:,:] = vc[0]
                dec_cpu_g.append(a)
            else:
                dl = []
                for id, vd in enumerate(vg):
                    d = numpy.empty_like(vd)
                    for ic, vc in enumerate(dec_cpu):
                        d[ic,:,:] = vc[ig][id]
                    dl.append(d)
                dec_cpu_g.append(dl)

        for i, (d1, d2) in enumerate(zip(dec_gpu, dec_cpu_g)):
            if i == 0:
                result1 = d1.flatten()
                result2 = d2.flatten()
            else:
                for d in d1:
                    result1 = numpy.concatenate((result1,d.flatten()))
                for d in d2:
                    result2 = numpy.concatenate((result2,d.flatten()))
        print('RMSE: {} \n'.format(rmse(result1, result2)))

        # Inverse Transform
        print('---------INVERSE 2D DWT---------')
        t = time.time()
        for _ in range(iterations):
            rec_cpu = [pywt.waverec2(d,wavelet=wavelet,mode='periodization') for d in dec_cpu]
        t = time.time()-t
        print('PyWavelets:\t\t\t\t{:.3f} ms'.format((t*1000.)/iterations))

        t = time.time()
        for _ in range(iterations):
            rec_gpu = pwt.idwt2d(dec_cpu_g)
        t = time.time()-t
        print('PycudaWaveletTransform:\t{:.3f} ms'.format((t*1000.)/iterations))

        rec_cpu_g = numpy.empty_like(rec_gpu)
        for ic, vc in enumerate(rec_cpu):
            rec_cpu_g[ic,:,:] = vc
        print('RMSE: {} '.format(rmse(rec_gpu,rec_cpu_g)))

    except Exception as e:
        tb = traceback.format_exc()
        print("%s",tb)


def main(argv):
    parser = argparse.ArgumentParser(description='2D DWT test of PycudaDWT')
    parser.add_argument('--dev_id', '-di', dest='dev_id', type=int, default=0, help='CUDA device id to use')
    parser.add_argument('--wavelet', '-w', dest='wavelet', type=str, default='haar', help='Wavelet type')
    parser.add_argument('--use_float32', dest='use_float32', action='store_true')
    parser.add_argument('--depth', '-d', dest='depth', type=int, default=1, help='Depth of transformation')
    parser.add_argument('--num_slices', '-ns', dest='num_slices', type=int, default=1, help='Number of 2D slices')
    parser.add_argument('--row_size', '-rs', dest='row_size', type=int, default=512, help='Size of rows')
    parser.add_argument('--col_size', '-cs', dest='col_size', type=int, default=512, help='Size of columns')
    parser.add_argument('--iterations', '-i', dest='iterations', type=int, default=20, help='Number of iterations for time profiling')
    parser.set_defaults(use_float32=False)
    args = parser.parse_args(argv)

    # Init CUDA
    cuda.init()
    cuda_device  = cuda.Device(args.dev_id)
    cuda_context = cuda_device.make_context()

    test2d(wavelet=args.wavelet, use_float32=args.use_float32, depth=args.depth, num_slices=args.num_slices, row_size=args.row_size, col_size=args.col_size, iterations=args.iterations)

    # Detach CUDA before leaving
    cuda_context.detach()

if __name__ == "__main__":
    main(sys.argv[1:])