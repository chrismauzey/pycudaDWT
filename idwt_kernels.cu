
#ifndef IDWT_KERNELS_CU
#define IDWT_KERNELS_CU

#define REC_LEN $rec_length

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
 * idwt_row
 *
 * Row-wise inverse discrete wavelet tranform
 *
 * @param inLo Input approximation array of type DTYPE
 * @param inHi Input detail array of type DTYPE
 * @param out Output array of type DTYPE
 * @param iRows The number of rows in the input arrays
 * @param lCols The number of columns in the approximation input array
 * @param hCols The number of columns in the detail input array
 */
__global__ void
idwt_row(DTYPE* inLo, DTYPE* inHi, DTYPE* out, int iRows, int lCols, int hCols)
{
    extern __shared__ DTYPE shr[];
    DTYPE recLo[REC_LEN] = $rec_lo;
    DTYPE recHi[REC_LEN] = $rec_hi;

    const int xid = blockDim.x*blockIdx.x + threadIdx.x;
    const int yid = blockDim.y*blockIdx.y + threadIdx.y;

    if(yid >= iRows) return;

    const int halfLen = REC_LEN/2;
    const int halfOff = halfLen/2;
    const int bufSize = halfLen + blockDim.x - 1;
    DTYPE* loBuf = &shr[2*bufSize*threadIdx.y];
    DTYPE* hiBuf = &shr[2*bufSize*threadIdx.y + bufSize];

    // populate shared memory buffer with input values
    int bufIt = bufSize/blockDim.x + ((bufSize%blockDim.x)?1:0);
    int off =  blockDim.x*blockIdx.x - halfOff;
    for(int i = 0; i < bufIt; ++i){
        int s = i*blockDim.x + threadIdx.x;
        if(s < bufSize){
            int cIdx = s + off;
            cIdx = periodic_idx(cIdx, hCols);
            loBuf[s] = inLo[lCols*yid + cIdx];
            hiBuf[s] = inHi[hCols*yid + cIdx];
        }
    }
    __syncthreads();

    // convolution
    if(xid < hCols){
        DTYPE loEven = 0.;
        DTYPE hiEven = 0.;
        DTYPE loOdd = 0.;
        DTYPE hiOdd = 0.;
        for(int i = 0; i < halfLen; ++i){
            DTYPE l = loBuf[threadIdx.x + i];
            DTYPE h = hiBuf[threadIdx.x + i];
            loEven += l*recLo[2*i+1];
            hiEven += h*recHi[2*i+1];
            loOdd += l*recLo[2*i];
            hiOdd += h*recHi[2*i];
        }

        if(halfLen%2 == 1){
            out[2*hCols*yid + 2*xid] = loEven + hiEven;
            out[2*hCols*yid + 2*xid + 1] =  loOdd + hiOdd;
        } else {
            out[2*hCols*yid + periodic_idx(2*xid - 1, 2*hCols)] = loEven + hiEven;
            out[2*hCols*yid + 2*xid] =  loOdd + hiOdd;
        }
    }

    return;
}

__global__ void
idwt_col(DTYPE *in, DTYPE* out, int nrows, int ncols)
{
//    extern __shared__ DTYPE shr[];
//    DTYPE recLo[REC_LEN] = $rec_lo;
//    DTYPE recHi[REC_LEN] = $rec_hi;

    return;
}

#endif //IDWT_KERNELS_CU
