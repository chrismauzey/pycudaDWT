"""
pycudaDWT

Written by Chris Mauzey 2018-02-28
"""

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import os.path
import string
import numpy
import pywt


class PycudaWaveletTransform:
    """
    PycudaWaveletTransform

    This class provides 1D and 2D discrete wavelet transforms (DWT) that run on CUDA-capable NVIDIA GPUs.
    The DWTs use periodic boundary extension, and the 2D transforms perform separable convolutions.
    This class only supports transforms for the following wavelet families:
        Haar
        Daubechies
        Symlets
        Coiflets
        Biorthogonal
        Reverse Biorthogonal
        Discrete Meyer (FIR Approximation)

    This package requires Numpy, PyCUDA, and PyWavelets (Tested with PyCUDA 2017.1 with CUDA 8.0)
    The module that uses the CuMedianFilter class must have initialized the
    PyCUDA driver and created a device context either automatically
        import pycuda.autoinit
    or manually
        import pycuda.driver
        pycuda.driver.init()
        dev = pycuda.driver.Device(gpuid)
        ctx = dev.make_context()
    """

    def __init__(self, wavelet='haar', use_float32=False):
        """
        Constructor

        Builds CUDA kernels used for forward and inverse DWTs using the specified wavelet family.

        Parameters
        ----------
            wavelet: str or pywt.Wavelet object, optional
                Either a string for the wavelet family or a pywt.Wavelet object ('haar' is the default)
            use_float32: boolean, optional
                Enable use of 32-bit floats for computing transforms (64-bit is the default)
        """

        self._use_float32 = use_float32
        self._dtype = numpy.float64
        dtype_str = 'double'
        if self._use_float32:
            self._dtype = numpy.float32
            dtype_str = 'float'

        self._wavelet = None
        if isinstance(wavelet, str):
            self._wavelet = pywt.Wavelet(wavelet)
        elif isinstance(wavelet, pywt.Wavelet):
            self._wavelet = pywt.Wavelet(wavelet.name)
        else:
            raise TypeError('wavelet is of an unsupported type')

        if self._wavelet.short_family_name not in ['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey']:
            raise RuntimeError('wavelet family not supported')

        nvcc_options = ["-Xcompiler", "-rdynamic", "-lineinfo", "--ptxas-options=-v", "-O3"]

        # Forward DWT filters
        # -------------------
        self._dec_length = self._wavelet.dec_len
        dec_lo = numpy.array(self._wavelet.dec_lo, dtype=self._dtype)[::-1]
        dec_hi = numpy.array(self._wavelet.dec_hi, dtype=self._dtype)[::-1]
        dec_lo_str = '{' + ','.join('{}'.format(x) for x in dec_lo) + '}'
        dec_hi_str = '{' + ','.join('{}'.format(x) for x in dec_hi) + '}'
        dwt_kernel_defines = {"dtype": dtype_str,
                              "dec_length": str(self._dec_length),
                              "dec_lo": dec_lo_str,
                              "dec_hi": dec_hi_str}

        dwt_kernel_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dwt_kernels.cu')
        with open(dwt_kernel_path, 'r') as dwt_kernel_file:
            dwt_kernel_source = string.Template(dwt_kernel_file.read()).substitute(dwt_kernel_defines)
        dwt_kernel_module = SourceModule(dwt_kernel_source, options=nvcc_options)
        self._dwt_row = dwt_kernel_module.get_function('dwt_row')
        self._dwt_col = dwt_kernel_module.get_function('dwt_col')
        self._dwt_row.set_cache_config(cuda.func_cache.PREFER_SHARED)
        self._dwt_col.set_cache_config(cuda.func_cache.PREFER_SHARED)

        # Inverse DWT filters
        # -------------------
        self._rec_length = self._wavelet.rec_len
        rec_lo = numpy.array(self._wavelet.rec_lo, dtype=self._dtype)[::-1]
        rec_hi = numpy.array(self._wavelet.rec_hi, dtype=self._dtype)[::-1]
        rec_lo_str = '{' + ','.join('{}'.format(x) for x in rec_lo) + '}'
        rec_hi_str = '{' + ','.join('{}'.format(x) for x in rec_hi) + '}'
        idwt_kernel_defines = {"dtype": dtype_str,
                               "rec_length": str(self._rec_length),
                               "rec_lo": rec_lo_str,
                               "rec_hi": rec_hi_str}

        idwt_kernel_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'idwt_kernels.cu')
        with open(idwt_kernel_path, 'r') as idwt_kernel_file:
            idwt_kernel_source = string.Template(idwt_kernel_file.read()).substitute(idwt_kernel_defines)
        idwt_kernel_module = SourceModule(idwt_kernel_source, options=nvcc_options)
        self._idwtRow = idwt_kernel_module.get_function('idwt_row')
        self._idwtCol = idwt_kernel_module.get_function('idwt_col')
        self._idwtRow.set_cache_config(cuda.func_cache.PREFER_SHARED)
        self._idwtCol.set_cache_config(cuda.func_cache.PREFER_SHARED)

    def dwt1d(self, input_array, depth=1):
        """
        dwt1d

        Perform a 1D forward discrete wavelet transform.
        If an array has multiple rows, this function will perform a batched transform.

        Parameters
        ----------
        input_array: Numpy array
            Input array array of data to be transformed.

            If the array is 1D, this function will perform a 1D forward DWT on the data.

            If the array is multidimensional, this function will perform a batched 1D forward DWT
            for all rows of the array.

            The rows being the last dimension of the array.
        depth: int, optional
            Depth level of transform; must be greater than or equal to 0 (1 is the default)

        Returns
        -------
        list of Numpy arrays
            A list of numpy arrays of the DWT coefficients either in float32 or float64,
            depending on whether use_float32 was set to True or False for PycudaWaveletTransform.
            The coefficients are organized as [c_lo, c_hi_depth, c_hi_depth-1, ... , c_hi_2, c_hi_1]
        """
        if not isinstance(input_array, numpy.ndarray):
            raise TypeError('invalid input array type')

        if depth < 0:
            raise RuntimeError('invalid depth value')
        elif depth == 0:
            return [input_array]

        cont_input_array = numpy.ascontiguousarray(input_array, dtype=self._dtype)
        in_shape = cont_input_array.shape
        row_size = int(in_shape[-1])
        num_rows = int(1)
        if len(in_shape) > 1:
            for i in in_shape[:-1]:
                num_rows *= i
        cont_input_array = cont_input_array.reshape([num_rows, row_size])

        # Get size of output arrays
        out_sizes = [int((row_size + 1) / 2 if row_size % 2 else (row_size / 2))]
        for i in range(depth - 1):
            ps = out_sizes[-1]
            ns = int((ps + 1) / 2 if ps % 2 else (ps / 2))
            out_sizes.append(int(ns))

        # Allocate device arrays
        approx_host_array = numpy.empty([num_rows, out_sizes[-1]], dtype=self._dtype)
        detail_host_arrays = []
        detail_device_arrays = []
        approx_device_arrays = []
        for s in out_sizes:
            detail_host_arrays.append(numpy.empty([num_rows, s], dtype=self._dtype))
            detail_device_arrays.append(cuda.mem_alloc(num_rows * s * self._dtype().itemsize))
            approx_device_arrays.append(cuda.mem_alloc(num_rows * s * self._dtype().itemsize))

        # Transform
        block = (256, 1, 1)
        grid = (int(out_sizes[0] / block[0]) + (1 if out_sizes[0] % block[0] else 0), num_rows, 1)
        shared_mem_size = block[1] * (self._dec_length + 2 * (block[0] - 1)) * self._dtype().itemsize
        self._dwt_row(cuda.In(cont_input_array), approx_device_arrays[0], detail_device_arrays[0],
                      numpy.int32(num_rows), numpy.int32(row_size), block=block, grid=grid, shared=shared_mem_size)

        for d in range(depth - 1):
            grid = (int(out_sizes[d] / block[0]) + (1 if out_sizes[d] % block[0] else 0), num_rows, 1)
            self._dwt_row(approx_device_arrays[d], approx_device_arrays[d + 1], detail_device_arrays[d + 1],
                          numpy.int32(num_rows), numpy.int32(out_sizes[d]), block=block, grid=grid,
                          shared=shared_mem_size)

        # Get results from device
        cuda.memcpy_dtoh(approx_host_array, approx_device_arrays[-1])
        if len(in_shape) > 1:
            new_shape = list(in_shape[:-1])
            new_shape.append(out_sizes[-1])
            approx_host_array = approx_host_array.reshape(new_shape)
        for i, (d, s) in enumerate(zip(detail_device_arrays, out_sizes)):
            cuda.memcpy_dtoh(detail_host_arrays[i], d)
            if len(in_shape) > 1:
                new_shape = list(in_shape[:-1])
                new_shape.append(s)
                detail_host_arrays[i] = detail_host_arrays[i].reshape(new_shape)
        results = [approx_host_array]
        for d in detail_host_arrays[::-1]:
            results.append(d)

        # Free device memory
        for a, d in zip(approx_device_arrays, detail_device_arrays):
            a.free()
            d.free()

        return results

    def idwt1d(self, input_list):
        """
        idwt1d

        Perform a 1D inverse discrete wavelet transform.
        If an array has multiple rows, this function will perform a batched transform.

        Parameters
        ----------
        input_list: list of Numpy arrays
            A list of numpy arrays of the DWT coefficients to be reconstructed.

            The coefficients are organized as [c_lo, c_hi_depth, c_hi_depth-1, ... , c_hi_2, c_hi_1].

            The detail row size can be one less than the approximation row size for each level.

            If the arrays are multidimensional, this function will perform a batched 1D inverse DWT
            for all rows of the arrays.  All arrays must have the same dimensions except for the last dimension.

            The rows being the last dimension of the array.

        Returns
        -------
        list of Numpy array
            A Numpy array of the reconstructed signal either in float32 or float64,
            depending on whether use_float32 was set to True or False for PycudaWaveletTransform.
        """

        if not isinstance(input_list, list):
            raise TypeError('invalid input_list type')
        if any([not isinstance(x, numpy.ndarray) for x in input_list]):
            raise TypeError('invalid input_list type')
        if len(input_list) > 1:
            approx_shape = list(input_list[0].shape)
            for d in input_list[1:]:
                detail_shape = list(d.shape)
                if detail_shape[-1] < approx_shape[-1]-1 or approx_shape[-1] < detail_shape[-1]:
                    raise RuntimeError('arrays have incompatible shapes')
                if len(detail_shape) > 1:
                    if approx_shape[:-1] != detail_shape[:-1]:
                        raise RuntimeError('arrays have incompatible shapes')
                approx_shape[-1] = 2 * detail_shape[-1]

        depth = len(input_list) - 1

        if depth == 0:
            return input_list[0]

        cont_input_list = [numpy.ascontiguousarray(i, dtype=self._dtype) for i in input_list]
        in_shape = cont_input_list[0].shape
        num_rows = int(1)
        if len(in_shape) > 1:
            for i in in_shape[:-1]:
                num_rows *= i
        cont_input_list = [i.reshape([num_rows, int(i.shape[-1])]) for i in cont_input_list]

        # Get size of input_list arrays
        in_sizes = []
        out_sizes = []
        for i in range(depth):
            s = input_list[i+1].shape[-1]
            in_sizes.append(int(s))
            out_sizes.append(int(2*s))

        # Allocate device arrays
        approx_host_array = numpy.empty([num_rows, out_sizes[-1]], dtype=self._dtype)
        approx_device_arrays = []
        for s in out_sizes:
            approx_device_arrays.append(cuda.mem_alloc(s * num_rows * self._dtype().itemsize))

        # Transform
        block = (256, 1, 1)
        grid = (int(in_sizes[0] / block[0]) + (1 if in_sizes[0] % block[0] else 0), num_rows, 1)
        shared_mem_size = block[1] * (2 * (int(self._dec_length / 2) + block[0] - 1)) * self._dtype().itemsize
        self._idwtRow(cuda.In(cont_input_list[0]), cuda.In(cont_input_list[1]), approx_device_arrays[0],
                      numpy.int32(num_rows), numpy.int32(in_shape[-1]), numpy.int32(in_sizes[0]),
                      block=block, grid=grid, shared=shared_mem_size)

        for d in range(depth - 1):
            grid = (int(in_sizes[d + 1] / block[0]) + (1 if in_sizes[d + 1] % block[0] else 0), num_rows, 1)
            self._idwtRow(approx_device_arrays[d], cuda.In(cont_input_list[d + 2]), approx_device_arrays[d + 1],
                          numpy.int32(num_rows), numpy.int32(out_sizes[d]), numpy.int32(in_sizes[d + 1]),
                          block=block, grid=grid, shared=shared_mem_size)

        # Get results from device
        cuda.memcpy_dtoh(approx_host_array, approx_device_arrays[-1])
        if len(in_shape) > 1:
            new_shape = list(in_shape[:-1])
            new_shape.append(approx_host_array.shape[-1])
            approx_host_array = approx_host_array.reshape(new_shape)

        # Free device memory
        for a in approx_device_arrays:
            a.free()

        return approx_host_array

    def dwt2d(self):
        pass

    def idwt2d(self):
        pass
