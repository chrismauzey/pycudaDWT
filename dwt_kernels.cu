
#ifndef DWT_KERNELS_CU
#define DWT_KERNELS_CU

#define DEC_LEN $dec_length

typedef $dtype DTYPE;

__device__ inline int
periodic_idx(int idx, int len)
{
    int newIdx = idx;
    if(newIdx < 0){
        newIdx -= len*(newIdx/len);
        newIdx += len;
    }
    if(newIdx > len-1){
        newIdx = newIdx%len;
    }
    return newIdx;
}

/**
 * dwt_row
 *
 * Row-wise forward discrete wavelet tranform
 *
 * @param in Input array of type DTYPE
 * @param outLo Output approximation array of type DTYPE
 * @param outHi Output detail array of type DTYPE
 * @param iRows The number of rows in the input array
 * @param iCols The number of columns in the input array
 */
__global__ void
dwt_row(DTYPE *in, DTYPE* outLo, DTYPE* outHi, int iRows, int iCols)
{
    extern __shared__ DTYPE shr[];
    DTYPE decLo[DEC_LEN] = $dec_lo;
    DTYPE decHi[DEC_LEN] = $dec_hi;

    const int xid = blockDim.x*blockIdx.x + threadIdx.x;
    const int yid = blockDim.y*blockIdx.y + threadIdx.y;

    if(yid >= iRows) return;

    const int filtOff = (DEC_LEN%2) ? (DEC_LEN/2) : (DEC_LEN/2 - 1);
    const int bufSize = DEC_LEN + 2*(blockDim.x - 1);
    DTYPE* buf = &shr[bufSize*threadIdx.y];

    // number of columns for output arrays
    int odd = (iCols&1);
    int oCols = (iCols+odd)/2;

    // populate shared memory buffer with input values
    int bufIt = bufSize/blockDim.x + ((bufSize%blockDim.x)?1:0);
    int off =  2*blockDim.x*blockIdx.x - filtOff;
    for(int i = 0; i < bufIt; ++i){
        int s = i*blockDim.x + threadIdx.x;
        if(s < bufSize){
            int cIdx = s + off;
            cIdx = periodic_idx(cIdx, iCols+odd);
            if(cIdx == iCols) --cIdx;  // When iCols is odd, extend the right-most edge by 1
            buf[s] = in[iCols*yid + cIdx];
        }
    }
    __syncthreads();

    // convolution
    if(xid < oCols){
        DTYPE lo = 0.;
        DTYPE hi = 0.;
        for(int i = 0; i < DEC_LEN; ++i){
            DTYPE v = buf[2*threadIdx.x + i];
            lo += v*decLo[i];
            hi += v*decHi[i];
        }

        outLo[oCols*yid + xid] = lo;
        outHi[oCols*yid + xid] = hi;
    }

    return;
}

__global__ void
dwt_col(DTYPE *in, DTYPE* out, int cRows, int oCols, int aRows, int aCols)
{
//    extern __shared__ DTYPE shr[];
//    DTYPE decLo[DEC_LEN] = $dec_lo;
//    DTYPE decHi[DEC_LEN] = $dec_hi;

    return;
}

#endif //DWT_KERNELS_CU
