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
        self._idwt_row = idwt_kernel_module.get_function('idwt_row')
        self._idwt_col = idwt_kernel_module.get_function('idwt_col')
        self._idwt_row.set_cache_config(cuda.func_cache.PREFER_SHARED)
        self._idwt_col.set_cache_config(cuda.func_cache.PREFER_SHARED)

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
        if len(in_shape) > 1:
            approx_host_array = numpy.empty([num_rows, out_sizes[-1]], dtype=self._dtype)
        else:
            approx_host_array = numpy.empty(out_sizes[-1], dtype=self._dtype)
        detail_host_arrays = []
        detail_device_arrays = []
        approx_device_arrays = []
        for s in out_sizes:
            if len(in_shape) > 1:
                detail_host_arrays.append(numpy.empty([num_rows, s], dtype=self._dtype))
            else:
                detail_host_arrays.append(numpy.empty(s, dtype=self._dtype))
            detail_device_arrays.append(cuda.mem_alloc(num_rows * s * self._dtype().itemsize))
            approx_device_arrays.append(cuda.mem_alloc(num_rows * s * self._dtype().itemsize))

        # Transform
        block = (256, 1, 1)
        grid_x = int(out_sizes[0] / block[0]) + (1 if out_sizes[0] % block[0] else 0)
        grid_y = int(num_rows / block[1]) + (1 if num_rows % block[1] else 0)
        grid = (grid_x, grid_y, 1)
        shared_mem_size = block[2] * block[1] * (self._dec_length + 2 * (block[0] - 1)) * self._dtype().itemsize
        self._dwt_row(cuda.In(cont_input_array), approx_device_arrays[0], detail_device_arrays[0],
                      numpy.int32(num_rows), numpy.int32(row_size), numpy.int32(1),
                      block=block, grid=grid, shared=shared_mem_size)

        for d in range(1, depth):
            grid_x = int(out_sizes[d] / block[0]) + (1 if out_sizes[d] % block[0] else 0)
            grid = (grid_x, grid_y, 1)
            self._dwt_row(approx_device_arrays[d-1], approx_device_arrays[d], detail_device_arrays[d],
                          numpy.int32(num_rows), numpy.int32(out_sizes[d-1]), numpy.int32(1),
                          block=block, grid=grid, shared=shared_mem_size)

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
        Numpy array
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
        if len(in_shape) > 1:
            approx_host_array = numpy.empty([num_rows, out_sizes[-1]], dtype=self._dtype)
        else:
            approx_host_array = numpy.empty(out_sizes[-1], dtype=self._dtype)
        approx_device_arrays = []
        for s in out_sizes:
            approx_device_arrays.append(cuda.mem_alloc(s * num_rows * self._dtype().itemsize))

        # Transform
        block = (256, 1, 1)
        grid_x = int(in_sizes[0] / block[0]) + (1 if in_sizes[0] % block[0] else 0)
        grid_y = int(num_rows / block[1]) + (1 if num_rows % block[1] else 0)
        grid = (grid_x, grid_y, 1)
        shared_mem_size = block[2] * block[1] * (2 * (int(self._dec_length / 2) + block[0] - 1)) * self._dtype().itemsize
        self._idwt_row(cuda.In(cont_input_list[0]), cuda.In(cont_input_list[1]), approx_device_arrays[0],
                       numpy.int32(num_rows), numpy.int32(in_shape[-1]), numpy.int32(in_sizes[0]), numpy.int32(1),
                       block=block, grid=grid, shared=shared_mem_size)

        for d in range(1, depth):
            grid_x = int(in_sizes[d] / block[0]) + (1 if in_sizes[d] % block[0] else 0)
            grid = (grid_x, grid_y, 1)
            self._idwt_row(approx_device_arrays[d-1], cuda.In(cont_input_list[d+1]), approx_device_arrays[d],
                           numpy.int32(num_rows), numpy.int32(out_sizes[d-1]), numpy.int32(in_sizes[d]), numpy.int32(1),
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

    def dwt2d(self, input_array, depth=1):
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

        Returns
        -------
        list of Numpy arrays
            A list of numpy arrays of the DWT coefficients either in float32 or float64,
            depending on whether use_float32 was set to True or False for PycudaWaveletTransform.
            The coefficients are organized as [c_lo_lo, (c_lo_hi_depth, c_hi_lo_depth, c_hi_hi_depth), ... , (c_lo_hi_1, c_hi_lo_1, c_hi_hi_1)]
        """
        if not isinstance(input_array, numpy.ndarray):
            raise TypeError('invalid input array type')

        if depth < 0:
            raise RuntimeError('invalid depth value')
        elif depth == 0:
            return [input_array]

        cont_input_array = numpy.ascontiguousarray(input_array, dtype=self._dtype)
        in_shape = cont_input_array.shape
        if len(in_shape) < 2:
            raise RuntimeError('input array must have 2 or more dimensions')

        num_cols = int(in_shape[-1])
        num_rows = int(in_shape[-2])
        num_slices = int(1)
        if len(in_shape) > 2:
            for i in in_shape[:-2]:
                num_slices *= i
            cont_input_array = cont_input_array.reshape([num_slices, num_rows, num_cols])

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

        # Allocate device arrays
        if len(in_shape) > 2:
            approx_host_array = numpy.empty([num_slices, out_rows[-1], out_cols[-1]], dtype=self._dtype)
        else:
            approx_host_array = numpy.empty([out_rows[-1], out_cols[-1]], dtype=self._dtype)
        row_approx_device_array = cuda.mem_alloc(num_slices * num_rows * out_cols[0] * self._dtype().itemsize)
        row_detail_device_array = cuda.mem_alloc(num_slices * num_rows * out_cols[0] * self._dtype().itemsize)
        detail_host_arrays = []
        output_device_arrays = []
        for r, c in zip(out_rows, out_cols):
            if len(in_shape) > 2:
                detail_host_arrays.append([numpy.empty([num_slices, r, c], dtype=self._dtype) for _ in range(3)])
            else:
                detail_host_arrays.append([numpy.empty([r, c], dtype=self._dtype) for _ in range(3)])
            output_device_arrays.append(dict(ll=cuda.mem_alloc(num_slices * r * c * self._dtype().itemsize),
                                             hl=cuda.mem_alloc(num_slices * r * c * self._dtype().itemsize),
                                             lh=cuda.mem_alloc(num_slices * r * c * self._dtype().itemsize),
                                             hh=cuda.mem_alloc(num_slices * r * c * self._dtype().itemsize)))

        # Transform
        row_block = (256, 1, 1)
        row_grid_x = int(out_cols[0] / row_block[0]) + (1 if out_cols[0] % row_block[0] else 0)
        row_grid_y = int(num_rows / row_block[1]) + (1 if num_rows % row_block[1] else 0)
        row_grid_z = int(num_slices / row_block[2]) + (1 if num_slices % row_block[2] else 0)
        row_grid = (row_grid_x, row_grid_y, row_grid_z)
        shared_mem_size = row_block[2] * row_block[1] * (self._dec_length + 2 * (row_block[0] - 1)) * self._dtype().itemsize
        self._dwt_row(cuda.In(cont_input_array), row_approx_device_array, row_detail_device_array,
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
            row_grid_y = int(out_rows[d-1] / row_block[1]) + (1 if out_rows[d-1] % row_block[1] else 0)
            row_grid = (row_grid_x, row_grid_y, row_grid_z)
            self._dwt_row(output_device_arrays[d-1]['ll'], row_approx_device_array, row_detail_device_array,
                          numpy.int32(out_rows[d-1]), numpy.int32(out_cols[d-1]), numpy.int32(num_slices),
                          block=row_block, grid=row_grid, shared=shared_mem_size)

            col_grid_x = int(out_cols[d] / col_block[0]) + (1 if out_cols[d] % col_block[0] else 0)
            col_grid_y = int(out_rows[d] / col_block[1]) + (1 if out_rows[d] % col_block[1] else 0)
            col_grid = (col_grid_x, col_grid_y, col_grid_z)
            self._dwt_col(row_approx_device_array, row_detail_device_array,
                          output_device_arrays[d]['ll'], output_device_arrays[d]['lh'],
                          output_device_arrays[d]['hl'], output_device_arrays[d]['hh'],
                          numpy.int32(out_rows[d-1]), numpy.int32(out_cols[d]), numpy.int32(num_slices),
                          block=col_block, grid=col_grid, shared=0)

        # Get results from device
        cuda.memcpy_dtoh(approx_host_array, output_device_arrays[-1]['ll'])
        if len(in_shape) > 2:
            new_shape = list(in_shape[:-2])
            new_shape.append(out_rows[-1])
            new_shape.append(out_cols[-1])
            approx_host_array = approx_host_array.reshape(new_shape)
        for i, (d, r, c) in enumerate(zip(output_device_arrays, out_rows, out_cols)):
            cuda.memcpy_dtoh(detail_host_arrays[i][0], d['lh'])
            cuda.memcpy_dtoh(detail_host_arrays[i][1], d['hl'])
            cuda.memcpy_dtoh(detail_host_arrays[i][2], d['hh'])
            if len(in_shape) > 2:
                new_shape = list(in_shape[:-2])
                new_shape.append(r)
                new_shape.append(c)
                detail_host_arrays[i][0] = detail_host_arrays[i][0].reshape(new_shape)
                detail_host_arrays[i][1] = detail_host_arrays[i][1].reshape(new_shape)
                detail_host_arrays[i][2] = detail_host_arrays[i][2].reshape(new_shape)
        results = [approx_host_array]
        for d in detail_host_arrays[::-1]:
            results.append(tuple(d))

        # Free device memory
        row_approx_device_array.free()
        row_detail_device_array.free()
        for d in output_device_arrays:
            d['ll'].free()
            d['hl'].free()
            d['lh'].free()
            d['hh'].free()

        return results

    def idwt2d(self, input_list):
        """
        idwt2d

        Perform a 2D inverse discrete wavelet transform.
        If the arrays have more than 2 dimensions, this function will perform a batched transform.

        Parameters
        ----------
        input_list: list of Numpy arrays
            A list of numpy arrays of the DWT coefficients to be reconstructed.

            The coefficients are organized as [c_lo_lo, (c_lo_hi_depth, c_hi_lo_depth, c_hi_hi_depth), ... , (c_lo_hi_1, c_hi_lo_1, c_hi_hi_1)].

            The detail array can have dimensions that are one less than that of the approximation array for each level.

            If the arrays have more than 2 dimensions, this function will perform a batched 2D inverse DWT
            for the last 2 dimensions of the arrays.  All arrays must have the same dimensions except for the last 2 dimensions.

        Returns
        -------
        Numpy array
            A Numpy array of the reconstructed signal either in float32 or float64,
            depending on whether use_float32 was set to True or False for PycudaWaveletTransform.
        """
        if not isinstance(input_list, list):
            raise TypeError('invalid input_list type')
        if not isinstance(input_list[0], numpy.ndarray):
            raise TypeError('invalid input_list type')
        if len(input_list) > 1:
            approx_shape = list(input_list[0].shape)
            for dl in input_list[1:]:
                if not isinstance(dl, list) and not isinstance(dl, tuple):
                    raise TypeError('invalid input_list type')
                elif len(dl) != 3:
                    raise TypeError('invalid input_list type')
                detail_shape = list(dl[0].shape)
                if list(dl[1].shape) != detail_shape or list(dl[2].shape) != detail_shape:
                    raise RuntimeError('arrays have incompatible shapes')
                if detail_shape[-1] < approx_shape[-1]-1 or approx_shape[-1] < detail_shape[-1]:
                    raise RuntimeError('arrays have incompatible shapes')
                if detail_shape[-2] < approx_shape[-2]-1 or approx_shape[-2] < detail_shape[-2]:
                    raise RuntimeError('arrays have incompatible shapes')
                if len(detail_shape) > 2:
                    if approx_shape[:-2] != detail_shape[:-2]:
                        raise RuntimeError('arrays have incompatible shapes')
                approx_shape[-1] = 2 * detail_shape[-1]
                approx_shape[-2] = 2 * detail_shape[-2]

        depth = len(input_list) - 1

        if depth == 0:
            return input_list[0]

        cont_input_list = [numpy.ascontiguousarray(input_list[0], dtype=self._dtype)]
        for i in input_list[1:]:
            cont_input_list.append([numpy.ascontiguousarray(c, dtype=self._dtype) for c in i])
        in_shape = cont_input_list[0].shape
        num_cols = int(in_shape[-1])
        num_rows = int(in_shape[-2])
        num_slices = int(1)
        if len(in_shape) > 2:
            for i in in_shape[:-2]:
                num_slices *= i
            cont_input_list[0] = cont_input_list[0].reshape([num_slices, num_rows, num_cols])
            for i, d in enumerate(cont_input_list[1:]):
                d_shape = d[0].shape
                cont_input_list[i+1] = [c.reshape(num_slices, d_shape[-2], d_shape[-1]) for c in d]

        # Get size of input_list arrays
        in_rows = []
        in_cols = []
        out_rows = []
        out_cols = []
        for i in range(depth):
            c = input_list[i+1][0].shape[-1]
            r = input_list[i+1][0].shape[-2]
            in_rows.append(int(r))
            in_cols.append(int(c))
            out_rows.append(int(2*r))
            out_cols.append(int(2*c))

        # Allocate device arrays
        if len(in_shape) > 2:
            approx_host_array = numpy.empty([num_slices, out_rows[-1], out_cols[-1]], dtype=self._dtype)
        else:
            approx_host_array = numpy.empty([out_rows[-1], out_cols[-1]], dtype=self._dtype)
        approx_device_array = cuda.mem_alloc(num_slices * out_rows[-1] * out_cols[-1] * self._dtype().itemsize)
        row_approx_device_array = cuda.mem_alloc(num_slices * out_rows[-1] * in_cols[-1] * self._dtype().itemsize)
        row_detail_device_array = cuda.mem_alloc(num_slices * out_rows[-1] * in_cols[-1] * self._dtype().itemsize)

        # Transform
        col_block = (256, 1, 1)
        col_grid_x = int(in_cols[0] / col_block[0]) + (1 if in_cols[0] % col_block[0] else 0)
        col_grid_y = int(in_rows[0] / col_block[1]) + (1 if in_rows[0] % col_block[1] else 0)
        col_grid_z = int(num_slices / col_block[2]) + (1 if num_slices % col_block[2] else 0)
        col_grid = (col_grid_x, col_grid_y, col_grid_z)
        self._idwt_col(cuda.In(cont_input_list[0]), cuda.In(cont_input_list[1][0]),
                       cuda.In(cont_input_list[1][1]), cuda.In(cont_input_list[1][2]),
                       row_approx_device_array, row_detail_device_array,
                       numpy.int32(num_rows), numpy.int32(num_cols),
                       numpy.int32(in_rows[0]), numpy.int32(in_cols[0]), numpy.int32(num_slices),
                       block=col_block, grid=col_grid, shared=0)

        row_block = (256, 1, 1)
        row_grid_x = int(in_cols[0] / row_block[0]) + (1 if in_cols[0] % row_block[0] else 0)
        row_grid_y = int(out_rows[0] / row_block[1]) + (1 if out_rows[0] % row_block[1] else 0)
        row_grid_z = int(num_slices / row_block[2]) + (1 if num_slices % row_block[2] else 0)
        row_grid = (row_grid_x, row_grid_y, row_grid_z)
        shared_mem_size = row_block[2] * row_block[1] * (2 * (int(self._dec_length / 2) + row_block[0] - 1)) * self._dtype().itemsize
        self._idwt_row(row_approx_device_array, row_detail_device_array, approx_device_array,
                       numpy.int32(out_rows[0]), numpy.int32(in_cols[0]), numpy.int32(in_cols[0]), numpy.int32(num_slices),
                       block=row_block, grid=row_grid, shared=shared_mem_size)

        for d in range(1, depth):
            col_grid_x = int(in_cols[d] / col_block[0]) + (1 if in_cols[d] % col_block[0] else 0)
            col_grid_y = int(in_rows[d] / col_block[1]) + (1 if in_rows[d] % col_block[1] else 0)
            col_grid = (col_grid_x, col_grid_y, col_grid_z)
            self._idwt_col(approx_device_array, cuda.In(cont_input_list[d+1][0]),
                           cuda.In(cont_input_list[d+1][1]), cuda.In(cont_input_list[d+1][2]),
                           row_approx_device_array, row_detail_device_array,
                           numpy.int32(out_rows[d-1]), numpy.int32(out_cols[d-1]),
                           numpy.int32(in_rows[d]), numpy.int32(in_cols[d]), numpy.int32(num_slices),
                           block=col_block, grid=col_grid, shared=0)

            row_grid_x = int(in_cols[d] / row_block[0]) + (1 if in_cols[d] % row_block[0] else 0)
            row_grid_y = int(out_rows[d] / row_block[1]) + (1 if out_rows[d] % row_block[1] else 0)
            row_grid = (row_grid_x, row_grid_y, row_grid_z)
            self._idwt_row(row_approx_device_array, row_detail_device_array, approx_device_array,
                           numpy.int32(out_rows[d]), numpy.int32(in_cols[d]), numpy.int32(in_cols[d]), numpy.int32(num_slices),
                           block=row_block, grid=row_grid, shared=shared_mem_size)

        # Get results from device
        cuda.memcpy_dtoh(approx_host_array, approx_device_array)
        if len(in_shape) > 2:
            new_shape = list(in_shape[:-2])
            new_shape.append(approx_host_array.shape[-2])
            new_shape.append(approx_host_array.shape[-1])
            approx_host_array = approx_host_array.reshape(new_shape)

        # Free device memory
        approx_device_array.free()
        row_approx_device_array.free()
        row_detail_device_array.free()

        return approx_host_array
