
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
 * @param nSlices The number of slices for 2D transforms
 */
__global__ void
idwt_row(DTYPE* inLo, DTYPE* inHi, DTYPE* out, int iRows, int lCols, int hCols, int nSlices)
{
    extern __shared__ DTYPE shr[];
    DTYPE recLo[REC_LEN] = $rec_lo;
    DTYPE recHi[REC_LEN] = $rec_hi;

    const int xid = blockDim.x*blockIdx.x + threadIdx.x;
    const int yid = blockDim.y*blockIdx.y + threadIdx.y;
    const int zid = blockDim.z*blockIdx.z + threadIdx.z;

    if(yid >= iRows || zid >= nSlices) return;

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
            loBuf[s] = inLo[iRows*lCols*zid + lCols*yid + cIdx];
            hiBuf[s] = inHi[iRows*hCols*zid + hCols*yid + cIdx];
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
            out[iRows*2*hCols*zid + 2*hCols*yid + 2*xid] = loEven + hiEven;
            out[iRows*2*hCols*zid + 2*hCols*yid + 2*xid + 1] =  loOdd + hiOdd;
        } else {
            out[iRows*2*hCols*zid + 2*hCols*yid + periodic_idx(2*xid - 1, 2*hCols)] = loEven + hiEven;
            out[iRows*2*hCols*zid + 2*hCols*yid + 2*xid] =  loOdd + hiOdd;
        }
    }

    return;
}


/**
 * idwt_col
 *
 * Column-wise inverse discrete wavelet tranform
 *
 * @param inLL Input column-wise approximation of row-wise approximation array of type DTYPE
 * @param inLH Input column-wise detail of row-wise approximation array of type DTYPE
 * @param inHL Input column-wise approximation of row-wise detail array of type DTYPE
 * @param inHH Input column-wise detail of row-wise detail array of type DTYPE
 * @param outLo Output row-wise approximation array of type DTYPE
 * @param outHi Output row-wise detail array of type DTYPE
 * @param lRows The number of rows in the approximation (LL) input array
 * @param lCols The number of columns in the approximation (LL) input array
 * @param hRows The number of rows in the detail (LH, HL, HH) input arrays
 * @param hCols The number of columns in the detail (LH, HL, HH) input arrays
 * @param nSlices The number of slices for 2D transforms
 */
__global__ void
idwt_col(DTYPE* inLL, DTYPE* inLH, DTYPE* inHL, DTYPE* inHH, DTYPE* outLo, DTYPE* outHi,
         int lRows, int lCols, int hRows, int hCols, int nSlices)
{
    DTYPE recLo[REC_LEN] = $rec_lo;
    DTYPE recHi[REC_LEN] = $rec_hi;

    const int xid = blockDim.x*blockIdx.x + threadIdx.x;
    const int yid = blockDim.y*blockIdx.y + threadIdx.y;
    const int zid = blockDim.z*blockIdx.z + threadIdx.z;

    if(xid >= hCols || zid >= nSlices) return;

    const int halfLen = REC_LEN/2;
    const int halfOff = halfLen/2;

    // convolution
    if(yid < hRows){
        DTYPE llEven = 0.;
        DTYPE lhEven = 0.;
        DTYPE hlEven = 0.;
        DTYPE hhEven = 0.;
        DTYPE llOdd = 0.;
        DTYPE lhOdd = 0.;
        DTYPE hlOdd = 0.;
        DTYPE hhOdd = 0.;
        for(int i = 0; i < halfLen; ++i){
            int rIdx = yid - halfOff + i;
            rIdx = periodic_idx(rIdx, hRows);
            DTYPE ll = inLL[lCols*rIdx + lRows*lCols*zid + xid];
            DTYPE lh = inLH[hCols*rIdx + hRows*hCols*zid + xid];
            DTYPE hl = inHL[hCols*rIdx + hRows*hCols*zid + xid];
            DTYPE hh = inHH[hCols*rIdx + hRows*hCols*zid + xid];
            llOdd += ll*recLo[2*i];
            llEven += ll*recLo[2*i+1];
            lhOdd += lh*recHi[2*i];
            lhEven += lh*recHi[2*i+1];
            hlOdd += hl*recLo[2*i];
            hlEven += hl*recLo[2*i+1];
            hhOdd += hh*recHi[2*i];
            hhEven += hh*recHi[2*i+1];
        }

        if(halfLen%2 == 1){
            outLo[hCols*(2*yid) + 2*hRows*hCols*zid + xid] = llEven + lhEven;
            outLo[hCols*(2*yid + 1) + 2*hRows*hCols*zid + xid] = llOdd + lhOdd;
            outHi[hCols*(2*yid) + 2*hRows*hCols*zid + xid] = hlEven + hhEven;
            outHi[hCols*(2*yid + 1) + 2*hRows*hCols*zid + xid] = hlOdd + hhOdd;
        } else {
            outLo[hCols*(periodic_idx(2*yid - 1, 2*hRows)) + 2*hRows*hCols*zid + xid] = llEven + lhEven;
            outLo[hCols*(2*yid) + 2*hRows*hCols*zid + xid] = llOdd + lhOdd;
            outHi[hCols*(periodic_idx(2*yid - 1, 2*hRows)) + 2*hRows*hCols*zid + xid] = hlEven + hhEven;
            outHi[hCols*(2*yid) + 2*hRows*hCols*zid + xid] = hlOdd + hhOdd;
        }

    }

    return;
}

#endif //IDWT_KERNELS_CU
