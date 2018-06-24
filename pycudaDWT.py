"""
pycudaDWT

Written by Chris Mauzey 2018-02-28
"""

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
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
        self._idwt_row = idwt_kernel_module.get_function('idwt_row')
        self._idwt_col = idwt_kernel_module.get_function('idwt_col')
        self._idwt_row.set_cache_config(cuda.func_cache.PREFER_SHARED)
        self._idwt_col.set_cache_config(cuda.func_cache.PREFER_SHARED)

    def dwt1d(self, input_array, depth=1, gpu_output=False, gpu_allocator=cuda.mem_alloc):
        """
        dwt1d

        Perform a 1D forward discrete wavelet transform.
        If an array has multiple rows, this function will perform a batched transform.

        Parameters
        ----------
        input_array: Numpy array or GPUArray object
            Input array array of data to be transformed.

            If the array is 1D, this function will perform a 1D forward DWT on the data.

            If the array is multidimensional, this function will perform a batched 1D forward DWT
            for all rows of the array.

            The rows being the last dimension of the array.
        depth: int, optional
            Depth level of transform; must be greater than or equal to 0 (1 is the default)
        gpu_output: bool, optional
            If True, then return the coefficients as a list of GPUArray objects. (False is the default)
        gpu_allocator: callable, optional
            Allocator used by GPUArray. (pycuda.driver.mem_alloc is the default)

        Returns
        -------
        list of Numpy arrays or GPUArray objects
            A list of numpy arrays of the DWT coefficients either in float32 or float64,
            depending on whether use_float32 was set to True or False for PycudaWaveletTransform.
            The coefficients are organized as [c_lo, c_hi_depth, c_hi_depth-1, ... , c_hi_2, c_hi_1]
        """
        if not isinstance(input_array, numpy.ndarray) and not isinstance(input_array, gpuarray.GPUArray):
            raise TypeError('invalid input array type')

        if depth < 0:
            raise RuntimeError('invalid depth value')
        elif depth == 0:
            return [input_array]

        in_shape = input_array.shape
        row_size = int(in_shape[-1])
        num_rows = int(1)
        if len(in_shape) > 1:
            for i in in_shape[:-1]:
                num_rows *= i

        # If input is in Numpy array, then copy it to a GPUArray object.
        if isinstance(input_array, numpy.ndarray):
            cont_input_array = numpy.ascontiguousarray(input_array, dtype=self._dtype)
            input_device_array = gpuarray.to_gpu(cont_input_array)
        else:
            if input_array.dtype != self._dtype:
                input_device_array = input_array.astype(self._dtype)
            else:
                input_device_array = input_array
            if not input_device_array.flags.c_contiguous:
                input_device_array = input_device_array.reshape(in_shape, order='C')

        # Get size of output arrays
        out_sizes = [int((row_size + 1) / 2 if row_size % 2 else (row_size / 2))]
        for i in range(depth - 1):
            ps = out_sizes[-1]
            ns = int((ps + 1) / 2 if ps % 2 else (ps / 2))
            out_sizes.append(int(ns))

        # Allocate device arrays
        detail_device_arrays = []
        approx_device_arrays = []
        for s in out_sizes:
            detail_device_arrays.append(gpuarray.GPUArray([num_rows, s], dtype=self._dtype, allocator=gpu_allocator))
            approx_device_arrays.append(gpuarray.GPUArray([num_rows, s], dtype=self._dtype, allocator=gpu_allocator))

        # Transform
        block = (256, 1, 1)
        grid_x = int(out_sizes[0] / block[0]) + (1 if out_sizes[0] % block[0] else 0)
        grid_y = int(num_rows / block[1]) + (1 if num_rows % block[1] else 0)
        grid = (grid_x, grid_y, 1)
        shared_mem_size = block[2] * block[1] * (self._dec_length + 2 * (block[0] - 1)) * self._dtype().itemsize
        self._dwt_row(input_device_array, approx_device_arrays[0], detail_device_arrays[0],
                      numpy.int32(num_rows), numpy.int32(row_size), numpy.int32(1),
                      block=block, grid=grid, shared=shared_mem_size)

        for d in range(1, depth):
            grid_x = int(out_sizes[d] / block[0]) + (1 if out_sizes[d] % block[0] else 0)
            grid = (grid_x, grid_y, 1)
            self._dwt_row(approx_device_arrays[d - 1], approx_device_arrays[d], detail_device_arrays[d],
                          numpy.int32(num_rows), numpy.int32(out_sizes[d - 1]), numpy.int32(1),
                          block=block, grid=grid, shared=shared_mem_size)

        # Get results from device
        approx_array = approx_device_arrays[-1] if gpu_output else approx_device_arrays[-1].get()
        if len(in_shape) > 1:
            new_shape = list(in_shape[:-1])
            new_shape.append(out_sizes[-1])
            approx_array = approx_array.reshape(new_shape)
        results = [approx_array]
        for i, (d, s) in enumerate(zip(detail_device_arrays[::-1], out_sizes[::-1])):
            detail_array = d if gpu_output else d.get()
            if len(in_shape) > 1:
                new_shape = list(in_shape[:-1])
                new_shape.append(s)
                detail_array = detail_array.reshape(new_shape)
            results.append(detail_array)

        return results

    def idwt1d(self, input_list, gpu_output=False, gpu_allocator=cuda.mem_alloc):
        """
        idwt1d

        Perform a 1D inverse discrete wavelet transform.
        If an array has multiple rows, this function will perform a batched transform.

        Parameters
        ----------
        input_list: list of Numpy arrays or GPUArray objects
            A list of numpy arrays of the DWT coefficients to be reconstructed.

            The coefficients are organized as [c_lo, c_hi_depth, c_hi_depth-1, ... , c_hi_2, c_hi_1].

            The detail row size can be one less than the approximation row size for each level.

            If the arrays are multidimensional, this function will perform a batched 1D inverse DWT
            for all rows of the arrays.  All arrays must have the same dimensions except for the last dimension.

            The rows being the last dimension of the array.
        gpu_output: bool, optional
            If True, then return the reconstructed signal as a GPUArray object. (False is the default)
        gpu_allocator: callable, optional
            Allocator used by GPUArray. (pycuda.driver.mem_alloc is the default)

        Returns
        -------
        Numpy array or GPUArray object
            An array of the reconstructed signal either in float32 or float64,
            depending on whether use_float32 was set to True or False for PycudaWaveletTransform.
        """

        if not isinstance(input_list, list):
            raise TypeError('invalid input_list type')
        if any([not isinstance(x, numpy.ndarray) and not isinstance(x, gpuarray.GPUArray) for x in input_list]):
            raise TypeError('invalid input_list type')
        if len(input_list) > 1:
            approx_shape = list(input_list[0].shape)
            for d in input_list[1:]:
                detail_shape = list(d.shape)
                if detail_shape[-1] < approx_shape[-1] - 1 or approx_shape[-1] < detail_shape[-1]:
                    raise RuntimeError('arrays have incompatible shapes')
                if len(detail_shape) > 1:
                    if approx_shape[:-1] != detail_shape[:-1]:
                        raise RuntimeError('arrays have incompatible shapes')
                approx_shape[-1] = 2 * detail_shape[-1]

        depth = len(input_list) - 1

        if depth == 0:
            return input_list[0]

        in_shape = input_list[0].shape
        num_rows = int(1)
        if len(in_shape) > 1:
            for i in in_shape[:-1]:
                num_rows *= i

        # If input is in Numpy arrays, then copy it to GPUArray objects.
        input_device_list = []
        for input_array in input_list:
            if isinstance(input_array, numpy.ndarray):
                cont_input_array = numpy.ascontiguousarray(input_array, dtype=self._dtype)
                input_device_array = gpuarray.to_gpu(cont_input_array)
            else:
                if input_array.dtype != self._dtype:
                    input_device_array = input_array.astype(self._dtype)
                else:
                    input_device_array = input_array
                if not input_device_array.flags.c_contiguous:
                    input_device_array = input_device_array.reshape(input_array.shape, order='C')
            input_device_list.append(input_device_array)

        # Get size of input_list arrays
        in_sizes = []
        out_sizes = []
        for i in range(depth):
            s = input_list[i + 1].shape[-1]
            in_sizes.append(int(s))
            out_sizes.append(int(2 * s))

        # Allocate device arrays
        approx_device_arrays = []
        for s in out_sizes:
            approx_device_arrays.append(gpuarray.GPUArray([num_rows, s], dtype=self._dtype, allocator=gpu_allocator))

        # Transform
        block = (256, 1, 1)
        grid_x = int(in_sizes[0] / block[0]) + (1 if in_sizes[0] % block[0] else 0)
        grid_y = int(num_rows / block[1]) + (1 if num_rows % block[1] else 0)
        grid = (grid_x, grid_y, 1)
        shared_mem_size = block[2]*block[1]*(2*(int(self._dec_length/2) + block[0] - 1))*self._dtype().itemsize
        self._idwt_row(input_device_list[0], input_device_list[1], approx_device_arrays[0],
                       numpy.int32(num_rows), numpy.int32(in_shape[-1]), numpy.int32(in_sizes[0]), numpy.int32(1),
                       block=block, grid=grid, shared=shared_mem_size)

        for d in range(1, depth):
            grid_x = int(in_sizes[d] / block[0]) + (1 if in_sizes[d] % block[0] else 0)
            grid = (grid_x, grid_y, 1)
            self._idwt_row(approx_device_arrays[d - 1], input_device_list[d + 1], approx_device_arrays[d],
                           numpy.int32(num_rows), numpy.int32(out_sizes[d - 1]), numpy.int32(in_sizes[d]),
                           numpy.int32(1),
                           block=block, grid=grid, shared=shared_mem_size)

        # Get results from device
        result = approx_device_arrays[-1] if gpu_output else approx_device_arrays[-1].get()
        if len(in_shape) > 1:
            new_shape = list(in_shape[:-1])
            new_shape.append(result.shape[-1])
            result = result.reshape(new_shape)

        return result

    def dwt2d(self, input_array, depth=1, gpu_output=False, gpu_allocator=cuda.mem_alloc):
        """
        dwt2d

        Perform a 2D forward discrete wavelet transform.
        If an array has more than 2 dimensions, this function will perform a batched transform.

        Parameters
        ----------
        input_array: Numpy array
            Input array array of data to be transformed.

            If the array is 2D, this function will perform a 2D forward DWT on the data.

            If the array has more than 2 dimensions, this function will perform a batched 2D forward DWT
            for the last 2 dimensions of the array.
        depth: int, optional
            Depth level of transform; must be greater than or equal to 0 (1 is the default)
        gpu_output: bool, optional
            If True, then return the coefficients as a list of GPUArray objects. (False is the default)
        gpu_allocator: callable, optional
            Allocator used by GPUArray. (pycuda.driver.mem_alloc is the default)

        Returns
        -------
        list of Numpy arrays or GPUArray objects
            A list of numpy arrays of the DWT coefficients either in float32 or float64,
            depending on whether use_float32 was set to True or False for PycudaWaveletTransform.
            The coefficients are organized as
            [c_lo_lo, (c_lo_hi_depth, c_hi_lo_depth, c_hi_hi_depth), ... , (c_lo_hi_1, c_hi_lo_1, c_hi_hi_1)]
        """
        if not isinstance(input_array, numpy.ndarray) and not isinstance(input_array, gpuarray.GPUArray):
            raise TypeError('invalid input array type')

        if depth < 0:
            raise RuntimeError('invalid depth value')
        elif depth == 0:
            return [input_array]

        in_shape = input_array.shape
        if len(in_shape) < 2:
            raise RuntimeError('input array must have 2 or more dimensions')

        num_cols = int(in_shape[-1])
        num_rows = int(in_shape[-2])
        num_slices = int(1)
        if len(in_shape) > 2:
            for i in in_shape[:-2]:
                num_slices *= i

        # Get size of output arrays
        out_cols = [int((num_cols + 1) / 2 if num_cols % 2 else (num_cols / 2))]
        out_rows = [int((num_rows + 1) / 2 if num_rows % 2 else (num_rows / 2))]
        for i in range(depth - 1):
            pc = out_cols[-1]
            nc = int((pc + 1) / 2 if pc % 2 else (pc / 2))
            out_cols.append(int(nc))
            pr = out_rows[-1]
            nr = int((pr + 1) / 2 if pr % 2 else (pr / 2))
            out_rows.append(int(nr))

        # If input is in Numpy array, then copy it to a GPUArray object.
        if isinstance(input_array, numpy.ndarray):
            cont_input_array = numpy.ascontiguousarray(input_array, dtype=self._dtype)
            input_device_array = gpuarray.to_gpu(cont_input_array)
        else:
            if input_array.dtype != self._dtype:
                input_device_array = input_array.astype(self._dtype)
            else:
                input_device_array = input_array
            if not input_device_array.flags.c_contiguous:
                input_device_array = input_device_array.reshape(in_shape, order='C')

        # Allocate device arrays
        row_approx_device_array = gpuarray.GPUArray([num_slices, num_rows, out_cols[0]],
                                                    dtype=self._dtype,
                                                    allocator=gpu_allocator)
        row_detail_device_array = gpuarray.GPUArray([num_slices, num_rows, out_cols[0]],
                                                    dtype=self._dtype,
                                                    allocator=gpu_allocator)
        output_device_arrays = []
        for r, c in zip(out_rows, out_cols):
            output_device_arrays.append(dict(ll=gpuarray.GPUArray([num_slices, r, c],
                                                                  dtype=self._dtype,
                                                                  allocator=gpu_allocator),
                                             hl=gpuarray.GPUArray([num_slices, r, c],
                                                                  dtype=self._dtype,
                                                                  allocator=gpu_allocator),
                                             lh=gpuarray.GPUArray([num_slices, r, c],
                                                                  dtype=self._dtype,
                                                                  allocator=gpu_allocator),
                                             hh=gpuarray.GPUArray([num_slices, r, c],
                                                                  dtype=self._dtype,
                                                                  allocator=gpu_allocator)))

        # Transform
        row_block = (256, 1, 1)
        row_grid_x = int(out_cols[0] / row_block[0]) + (1 if out_cols[0] % row_block[0] else 0)
        row_grid_y = int(num_rows / row_block[1]) + (1 if num_rows % row_block[1] else 0)
        row_grid_z = int(num_slices / row_block[2]) + (1 if num_slices % row_block[2] else 0)
        row_grid = (row_grid_x, row_grid_y, row_grid_z)
        shared_mem_size = row_block[2]*row_block[1]*(self._dec_length + 2*(row_block[0] - 1))*self._dtype().itemsize
        self._dwt_row(input_device_array, row_approx_device_array, row_detail_device_array,
                      numpy.int32(num_rows), numpy.int32(num_cols), numpy.int32(num_slices),
                      block=row_block, grid=row_grid, shared=shared_mem_size)

        col_block = (256, 1, 1)
        col_grid_x = int(out_cols[0] / col_block[0]) + (1 if out_cols[0] % col_block[0] else 0)
        col_grid_y = int(out_rows[0] / col_block[1]) + (1 if out_rows[0] % col_block[1] else 0)
        col_grid_z = int(num_slices / col_block[2]) + (1 if num_slices % col_block[2] else 0)
        col_grid = (col_grid_x, col_grid_y, col_grid_z)
        self._dwt_col(row_approx_device_array, row_detail_device_array,
                      output_device_arrays[0]['ll'], output_device_arrays[0]['lh'],
                      output_device_arrays[0]['hl'], output_device_arrays[0]['hh'],
                      numpy.int32(num_rows), numpy.int32(out_cols[0]), numpy.int32(num_slices),
                      block=col_block, grid=col_grid, shared=0)

        for d in range(1, depth):
            row_grid_x = int(out_cols[d] / row_block[0]) + (1 if out_cols[d] % row_block[0] else 0)
            row_grid_y = int(out_rows[d - 1] / row_block[1]) + (1 if out_rows[d - 1] % row_block[1] else 0)
            row_grid = (row_grid_x, row_grid_y, row_grid_z)
            self._dwt_row(output_device_arrays[d - 1]['ll'], row_approx_device_array, row_detail_device_array,
                          numpy.int32(out_rows[d - 1]), numpy.int32(out_cols[d - 1]), numpy.int32(num_slices),
                          block=row_block, grid=row_grid, shared=shared_mem_size)

            col_grid_x = int(out_cols[d] / col_block[0]) + (1 if out_cols[d] % col_block[0] else 0)
            col_grid_y = int(out_rows[d] / col_block[1]) + (1 if out_rows[d] % col_block[1] else 0)
            col_grid = (col_grid_x, col_grid_y, col_grid_z)
            self._dwt_col(row_approx_device_array, row_detail_device_array,
                          output_device_arrays[d]['ll'], output_device_arrays[d]['lh'],
                          output_device_arrays[d]['hl'], output_device_arrays[d]['hh'],
                          numpy.int32(out_rows[d - 1]), numpy.int32(out_cols[d]), numpy.int32(num_slices),
                          block=col_block, grid=col_grid, shared=0)

        # Get results from device
        approx_array = output_device_arrays[-1]['ll'] if gpu_output else output_device_arrays[-1]['ll'].get()
        if len(in_shape) > 2:
            new_shape = list(in_shape[:-2])
            new_shape.append(out_rows[-1])
            new_shape.append(out_cols[-1])
            approx_array = approx_array.reshape(new_shape)
        results = [approx_array]
        for d, r, c in zip(output_device_arrays[::-1], out_rows[::-1], out_cols[::-1]):
            detail_lh_array = d['lh'] if gpu_output else d['lh'].get()
            detail_hl_array = d['hl'] if gpu_output else d['hl'].get()
            detail_hh_array = d['hh'] if gpu_output else d['hh'].get()
            if len(in_shape) > 2:
                new_shape = list(in_shape[:-2])
                new_shape.append(r)
                new_shape.append(c)
                detail_lh_array = detail_lh_array.reshape(new_shape)
                detail_hl_array = detail_hl_array.reshape(new_shape)
                detail_hh_array = detail_hh_array.reshape(new_shape)
            results.append((detail_lh_array, detail_hl_array, detail_hh_array))

        return results

    def idwt2d(self, input_list, gpu_output=False, gpu_allocator=cuda.mem_alloc):
        """
        idwt2d

        Perform a 2D inverse discrete wavelet transform.
        If the arrays have more than 2 dimensions, this function will perform a batched transform.

        Parameters
        ----------
        input_list: list of Numpy arrays
            A list of numpy arrays of the DWT coefficients to be reconstructed.

            The coefficients are organized as
            [c_lo_lo, (c_lo_hi_depth, c_hi_lo_depth, c_hi_hi_depth), ... , (c_lo_hi_1, c_hi_lo_1, c_hi_hi_1)].

            The detail array can have dimensions that are one less than that of the approximation array for each level.

            If the arrays have more than 2 dimensions, this function will perform a batched 2D inverse DWT
            for the last 2 dimensions of the arrays.  All arrays must have the same dimensions
            except for the last 2 dimensions.
        gpu_output: bool, optional
            If True, then return the reconstructed signal as a GPUArray object. (False is the default)
        gpu_allocator: callable, optional
            Allocator used by GPUArray. (pycuda.driver.mem_alloc is the default)
        Returns
        -------
        Numpy array or GPUArray object
            A Numpy array of the reconstructed signal either in float32 or float64,
            depending on whether use_float32 was set to True or False for PycudaWaveletTransform.
        """
        if not isinstance(input_list, list):
            raise TypeError('invalid input_list type')
        if not isinstance(input_list[0], numpy.ndarray) and not isinstance(input_list[0], gpuarray.GPUArray):
            raise TypeError('invalid input_list type')
        if len(input_list) > 1:
            approx_shape = list(input_list[0].shape)
            for dl in input_list[1:]:
                if not isinstance(dl, list) and not isinstance(dl, tuple):
                    raise TypeError('invalid input_list type')
                elif len(dl) != 3:
                    raise TypeError('invalid input_list type')
                if any([not isinstance(x, numpy.ndarray) and not isinstance(x, gpuarray.GPUArray) for x in dl]):
                    raise TypeError('invalid input_list type')
                detail_shape = list(dl[0].shape)
                if list(dl[1].shape) != detail_shape or list(dl[2].shape) != detail_shape:
                    raise RuntimeError('arrays have incompatible shapes')
                if detail_shape[-1] < approx_shape[-1] - 1 or approx_shape[-1] < detail_shape[-1]:
                    raise RuntimeError('arrays have incompatible shapes')
                if detail_shape[-2] < approx_shape[-2] - 1 or approx_shape[-2] < detail_shape[-2]:
                    raise RuntimeError('arrays have incompatible shapes')
                if len(detail_shape) > 2:
                    if approx_shape[:-2] != detail_shape[:-2]:
                        raise RuntimeError('arrays have incompatible shapes')
                approx_shape[-1] = 2 * detail_shape[-1]
                approx_shape[-2] = 2 * detail_shape[-2]

        depth = len(input_list) - 1

        if depth == 0:
            return input_list[0]

        in_shape = input_list[0].shape
        num_cols = int(in_shape[-1])
        num_rows = int(in_shape[-2])
        num_slices = int(1)
        if len(in_shape) > 2:
            for i in in_shape[:-2]:
                num_slices *= i

        # Get size of input_list arrays
        in_rows = []
        in_cols = []
        out_rows = []
        out_cols = []
        for i in range(depth):
            c = input_list[i + 1][0].shape[-1]
            r = input_list[i + 1][0].shape[-2]
            in_rows.append(int(r))
            in_cols.append(int(c))
            out_rows.append(int(2 * r))
            out_cols.append(int(2 * c))

        # Allocate device arrays
        approx_device_array = gpuarray.GPUArray([num_slices, out_rows[-1], out_cols[-1]], dtype=self._dtype,
                                                allocator=gpu_allocator)
        row_approx_device_array = gpuarray.GPUArray([num_slices, out_rows[-1], in_cols[-1]], dtype=self._dtype,
                                                    allocator=gpu_allocator)
        row_detail_device_array = gpuarray.GPUArray([num_slices, out_rows[-1], in_cols[-1]], dtype=self._dtype,
                                                    allocator=gpu_allocator)

        # If input is in Numpy arrays, then copy it to GPUArray objects.
        input_device_list = []
        for input_index, input_array in enumerate(input_list):
            if input_index == 0:
                if isinstance(input_array, numpy.ndarray):
                    cont_input_array = numpy.ascontiguousarray(input_array, dtype=self._dtype)
                    input_device_array = gpuarray.to_gpu(cont_input_array)
                else:
                    if input_array.dtype != self._dtype:
                        input_device_array = input_array.astype(self._dtype)
                    else:
                        input_device_array = input_array
                    if not input_device_array.flags.c_contiguous:
                        input_device_array = input_device_array.reshape(input_array.shape, order='C')
                input_device_list.append(input_device_array)
            else:
                detail_list = []
                for detail_array in input_array:
                    if isinstance(detail_array, numpy.ndarray):
                        cont_input_array = numpy.ascontiguousarray(detail_array, dtype=self._dtype)
                        input_device_array = gpuarray.to_gpu(cont_input_array)
                    else:
                        if detail_array.dtype != self._dtype:
                            input_device_array = detail_array.astype(self._dtype)
                        else:
                            input_device_array = detail_array
                        if not input_device_array.flags.c_contiguous:
                            input_device_array = input_device_array.reshape(detail_array.shape, order='C')
                    detail_list.append(input_device_array)
                input_device_list.append(detail_list)

        # Transform
        col_block = (256, 1, 1)
        col_grid_x = int(in_cols[0] / col_block[0]) + (1 if in_cols[0] % col_block[0] else 0)
        col_grid_y = int(in_rows[0] / col_block[1]) + (1 if in_rows[0] % col_block[1] else 0)
        col_grid_z = int(num_slices / col_block[2]) + (1 if num_slices % col_block[2] else 0)
        col_grid = (col_grid_x, col_grid_y, col_grid_z)
        self._idwt_col(input_device_list[0], input_device_list[1][0],
                       input_device_list[1][1], input_device_list[1][2],
                       row_approx_device_array, row_detail_device_array,
                       numpy.int32(num_rows), numpy.int32(num_cols),
                       numpy.int32(in_rows[0]), numpy.int32(in_cols[0]), numpy.int32(num_slices),
                       block=col_block, grid=col_grid, shared=0)

        row_block = (256, 1, 1)
        row_grid_x = int(in_cols[0] / row_block[0]) + (1 if in_cols[0] % row_block[0] else 0)
        row_grid_y = int(out_rows[0] / row_block[1]) + (1 if out_rows[0] % row_block[1] else 0)
        row_grid_z = int(num_slices / row_block[2]) + (1 if num_slices % row_block[2] else 0)
        row_grid = (row_grid_x, row_grid_y, row_grid_z)
        shared_mem_size = row_block[2] * row_block[1] * (
                2 * (int(self._dec_length / 2) + row_block[0] - 1)) * self._dtype().itemsize
        self._idwt_row(row_approx_device_array, row_detail_device_array, approx_device_array,
                       numpy.int32(out_rows[0]), numpy.int32(in_cols[0]), numpy.int32(in_cols[0]),
                       numpy.int32(num_slices),
                       block=row_block, grid=row_grid, shared=shared_mem_size)

        for d in range(1, depth):
            col_grid_x = int(in_cols[d] / col_block[0]) + (1 if in_cols[d] % col_block[0] else 0)
            col_grid_y = int(in_rows[d] / col_block[1]) + (1 if in_rows[d] % col_block[1] else 0)
            col_grid = (col_grid_x, col_grid_y, col_grid_z)
            self._idwt_col(approx_device_array, input_device_list[d + 1][0],
                           input_device_list[d + 1][1], input_device_list[d + 1][2],
                           row_approx_device_array, row_detail_device_array,
                           numpy.int32(out_rows[d - 1]), numpy.int32(out_cols[d - 1]),
                           numpy.int32(in_rows[d]), numpy.int32(in_cols[d]), numpy.int32(num_slices),
                           block=col_block, grid=col_grid, shared=0)

            row_grid_x = int(in_cols[d] / row_block[0]) + (1 if in_cols[d] % row_block[0] else 0)
            row_grid_y = int(out_rows[d] / row_block[1]) + (1 if out_rows[d] % row_block[1] else 0)
            row_grid = (row_grid_x, row_grid_y, row_grid_z)
            self._idwt_row(row_approx_device_array, row_detail_device_array, approx_device_array,
                           numpy.int32(out_rows[d]), numpy.int32(in_cols[d]), numpy.int32(in_cols[d]),
                           numpy.int32(num_slices),
                           block=row_block, grid=row_grid, shared=shared_mem_size)

        # Get results from device
        approx_array = approx_device_array if gpu_output else approx_device_array.get()
        if len(in_shape) > 2:
            new_shape = list(in_shape[:-2])
            new_shape.append(approx_array.shape[-2])
            new_shape.append(approx_array.shape[-1])
            approx_array = approx_array.reshape(new_shape)

        return approx_array
