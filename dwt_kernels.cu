
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
 * @param nSlices The number of slices for 2D transforms
 */
__global__ void
dwt_row(DTYPE *in, DTYPE* outLo, DTYPE* outHi, int iRows, int iCols, int nSlices)
{
    extern __shared__ DTYPE shr[];
    DTYPE decLo[DEC_LEN] = $dec_lo;
    DTYPE decHi[DEC_LEN] = $dec_hi;

    const int xid = blockDim.x*blockIdx.x + threadIdx.x;
    const int yid = blockDim.y*blockIdx.y + threadIdx.y;
    const int zid = blockDim.z*blockIdx.z + threadIdx.z;

    if(yid >= iRows || zid >= nSlices) return;

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
            buf[s] = in[iRows*iCols*zid + iCols*yid + cIdx];
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

        outLo[iRows*oCols*zid + oCols*yid + xid] = lo;
        outHi[iRows*oCols*zid + oCols*yid + xid] = hi;
    }

    return;
}

/**
 * dwt_col
 *
 * Column-wise forward discrete wavelet tranform
 *
 * @param inLo Input row-wise approximation array of type DTYPE
 * @param inHi Input row-wise detail array of type DTYPE
 * @param outLL Output column-wise approximation of row-wise approximation array of type DTYPE
 * @param outLH Output column-wise detail of row-wise approximation array of type DTYPE
 * @param outHL Output column-wise approximation of row-wise detail array of type DTYPE
 * @param outHH Output column-wise detail of row-wise detail array of type DTYPE
 * @param iRows The number of rows in the input arrays
 * @param iCols The number of columns in the input arrays
 * @param nSlices The number of slices for 2D transforms
 */
__global__ void
dwt_col(DTYPE* inLo, DTYPE* inHi, DTYPE* outLL, DTYPE* outLH, DTYPE* outHL, DTYPE* outHH,
        int iRows, int iCols, int nSlices)
{
    DTYPE decLo[DEC_LEN] = $dec_lo;
    DTYPE decHi[DEC_LEN] = $dec_hi;

    const int xid = blockDim.x*blockIdx.x + threadIdx.x;
    const int yid = blockDim.y*blockIdx.y + threadIdx.y;
    const int zid = blockDim.z*blockIdx.z + threadIdx.z;

    if(xid >= iCols || zid >= nSlices) return;

    // number of rows for output arrays
    int odd = (iRows&1);
    int oRows = (iRows+odd)/2;

    const int filtOff = (DEC_LEN%2) ? (DEC_LEN/2) : (DEC_LEN/2 - 1);

    // convolution
    if(yid < oRows){
        DTYPE ll = 0.;
        DTYPE lh = 0.;
        DTYPE hl = 0.;
        DTYPE hh = 0.;
        for(int i = 0; i < DEC_LEN; ++i){
            int rIdx = 2*yid - filtOff + i;
            rIdx = periodic_idx(rIdx, iRows+odd);
            if(rIdx == iRows) --rIdx;  // When iRows is odd, extend the bottom-most edge by 1
            DTYPE vl = inLo[iCols*rIdx + iRows*iCols*zid + xid];
            DTYPE vh = inHi[iCols*rIdx + iRows*iCols*zid + xid];
            ll += vl*decLo[i];
            lh += vl*decHi[i];
            hl += vh*decLo[i];
            hh += vh*decHi[i];
        }

        outLL[iCols*yid + oRows*iCols*zid + xid] = ll;
        outLH[iCols*yid + oRows*iCols*zid + xid] = lh;
        outHL[iCols*yid + oRows*iCols*zid + xid] = hl;
        outHH[iCols*yid + oRows*iCols*zid + xid] = hh;
    }

    return;
}

#endif //DWT_KERNELS_CU
