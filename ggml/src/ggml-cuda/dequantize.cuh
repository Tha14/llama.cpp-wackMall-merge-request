#include "common.cuh"
#include "convert.cuh"
#include "turbo-quant.cuh"

static __device__ __forceinline__ void dequantize_q1_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q1_0 * x = (const block_q1_0 *) vx;

    const float d = x[ib].d;

    const int bit_index_0 = iqs;
    const int bit_index_1 = iqs + 1;

    const int byte_index_0 = bit_index_0 / 8;
    const int bit_offset_0 = bit_index_0 % 8;

    const int byte_index_1 = bit_index_1 / 8;
    const int bit_offset_1 = bit_index_1 % 8;

    // Extract bits: 1 = +d, 0 = -d (branchless)
    const int bit_0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 1;
    const int bit_1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 1;

    v.x = (2*bit_0 - 1) * d;
    v.y = (2*bit_1 - 1) * d;
}

static __device__ __forceinline__ void dequantize_q2_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q2_0 * x = (const block_q2_0 *) vx;

    const float d = x[ib].d;

    // Q2_0: 2 bits per element, 4 elements per byte.
    // Stored code c in {0,1,2,3} maps to symbol s = c - 1 in {-1, 0, +1, +2}.
    const int byte_index_0 = iqs / 4;
    const int bit_offset_0 = (iqs % 4) * 2;

    const int byte_index_1 = (iqs + 1) / 4;
    const int bit_offset_1 = ((iqs + 1) % 4) * 2;

    const int c0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 0x3;
    const int c1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 0x3;

    v.x = (c0 - 1) * d;
    v.y = (c1 - 1) * d;
}

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

// Turbo4: 4-bit PolarQuant (nibble packed), block size 128
// iqs is the element index within the block (even), produces elements iqs and iqs+1
static __device__ __forceinline__ void dequantize_turbo4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_turbo4_0 * x = (const block_turbo4_0 *) vx;
    const float norm = __half2float(x[ib].norm);
    v.x = turbo4_dequant_element(&x[ib], iqs + 0, norm);
    v.y = turbo4_dequant_element(&x[ib], iqs + 1, norm);
}

// Turbo3: 3-bit PolarQuant (2-bit qs + 1-bit sign), block size 32
// iqs is the element index within the block (even), produces elements iqs and iqs+1
static __device__ __forceinline__ void dequantize_turbo3_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_turbo3_0 * x = (const block_turbo3_0 *) vx;
    const float norm = __half2float(x[ib].norm);
    v.x = turbo3_dequant_element(&x[ib], iqs + 0, norm);
    v.y = turbo3_dequant_element(&x[ib], iqs + 1, norm);
}

// Turbo2: 2-bit PolarQuant (2-bit qs only, no sign), block size 32
static __device__ __forceinline__ void dequantize_turbo2_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_turbo2_0 * x = (const block_turbo2_0 *) vx;
    const float norm = __half2float(x[ib].norm);
    v.x = turbo2_dequant_element(&x[ib], iqs + 0, norm);
    v.y = turbo2_dequant_element(&x[ib], iqs + 1, norm);
}

// PlanarQuant / IsoQuant dequantize for flash attention
#include "planar-iso-constants.cuh"

static __constant__ float dq_centroids_3bit[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};
static __constant__ float dq_centroids_4bit[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
};

// Helper: unpack 3-bit index from block
static __device__ __forceinline__ uint8_t unpack_3bit(const block_planar3_0 * blk, int j) {
    uint8_t low = (blk->qs[j/4] >> ((j%4)*2)) & 0x3;
    uint8_t hi = (blk->signs[j/8] >> (j%8)) & 0x1;
    return low | (hi << 2);
}

// Helper: unpack 4-bit index from turbo4 block
static __device__ __forceinline__ uint8_t unpack_4bit(const block_turbo4_0 * blk, int j) {
    return (blk->qs[j/2] >> ((j%2)*4)) & 0xF;
}

// Planar3: 2D Givens inverse rotation - each pair is independent
static __device__ __forceinline__ void dequantize_planar3_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_planar3_0 * x = (const block_planar3_0 *) vx;
    const float norm = __half2float(x[ib].norm);

    // iqs must be even (pairs)
    float q0 = dq_centroids_3bit[unpack_3bit(&x[ib], iqs)];
    float q1 = dq_centroids_3bit[unpack_3bit(&x[ib], iqs + 1)];

    // Inverse Givens: pair index = iqs/2
    int p = iqs / 2;
    float c = PI_COS[p];
    float s = PI_SIN[p];
    v.x = ( c * q0 + s * q1) * norm;
    v.y = (-s * q0 + c * q1) * norm;
}

// Iso3: quaternion 4D inverse rotation - needs full 4-element group
static __device__ __forceinline__ void dequantize_iso3_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_iso3_0 * x = (const block_iso3_0 *) vx;
    const float norm = __half2float(x[ib].norm);

    // Quaternion group = iqs/4 (each group has 4 elements)
    int g = iqs / 4;
    int offset = iqs % 4;  // 0 or 2 within the group

    // Unpack all 4 elements of the group
    float qvals[4];
    for (int c = 0; c < 4; c++) {
        qvals[c] = dq_centroids_3bit[unpack_3bit((const block_planar3_0 *)&x[ib], g*4 + c)];
    }

    // Inverse quaternion: conj(q_L) * v
    float qw = PI_QW[g], qx = -PI_QX[g], qy = -PI_QY[g], qz = -PI_QZ[g];
    float rw = qw*qvals[0] - qx*qvals[1] - qy*qvals[2] - qz*qvals[3];
    float rx = qw*qvals[1] + qx*qvals[0] + qy*qvals[3] - qz*qvals[2];
    float ry = qw*qvals[2] - qx*qvals[3] + qy*qvals[0] + qz*qvals[1];
    float rz = qw*qvals[3] + qx*qvals[2] - qy*qvals[1] + qz*qvals[0];

    float results[4] = {rw, rx, ry, rz};
    v.x = results[offset] * norm;
    v.y = results[offset + 1] * norm;
}

// Planar4: 2D Givens + 4-bit
static __device__ __forceinline__ void dequantize_planar4_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_planar4_0 * x = (const block_planar4_0 *) vx;
    const float norm = __half2float(x[ib].norm);

    float q0 = dq_centroids_4bit[unpack_4bit(&x[ib], iqs)];
    float q1 = dq_centroids_4bit[unpack_4bit(&x[ib], iqs + 1)];

    int p = iqs / 2;
    float c = PI_COS[p];
    float s = PI_SIN[p];
    v.x = ( c * q0 + s * q1) * norm;
    v.y = (-s * q0 + c * q1) * norm;
}

// Iso4: quaternion + 4-bit
static __device__ __forceinline__ void dequantize_iso4_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_iso4_0 * x = (const block_iso4_0 *) vx;
    const float norm = __half2float(x[ib].norm);

    int g = iqs / 4;
    int offset = iqs % 4;

    float qvals[4];
    for (int c = 0; c < 4; c++) {
        qvals[c] = dq_centroids_4bit[unpack_4bit(&x[ib], g*4 + c)];
    }

    float qw = PI_QW[g], qx = -PI_QX[g], qy = -PI_QY[g], qz = -PI_QZ[g];
    float rw = qw*qvals[0] - qx*qvals[1] - qy*qvals[2] - qz*qvals[3];
    float rx = qw*qvals[1] + qx*qvals[0] + qy*qvals[3] - qz*qvals[2];
    float ry = qw*qvals[2] - qx*qvals[3] + qy*qvals[0] + qz*qvals[1];
    float rz = qw*qvals[3] + qx*qvals[2] - qy*qvals[1] + qz*qvals[0];

    float results[4] = {rw, rx, ry, rz};
    v.x = results[offset] * norm;
    v.y = results[offset + 1] * norm;
}

//================================== k-quants

// Each call dequantizes one super-block of QK_K values into y using the
// thread layout of the caller: 32 threads for q4_K, 64 threads otherwise.

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q2_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q2_K * x = (const block_q2_K *) vx;

    const int64_t n   = tid/32;
    const int64_t l   = tid - 32*n;
    const int64_t is  = 8*n + l/16;

    const uint8_t q = x[ib].qs[32*n + l];
    dst_t * y = yy + 128*n;

    float dall = __low2half(x[ib].dm);
    float dmin = __high2half(x[ib].dm);
    y[l+ 0] = ggml_cuda_cast<dst_t>(dall * (x[ib].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[ib].scales[is+0] >> 4));
    y[l+32] = ggml_cuda_cast<dst_t>(dall * (x[ib].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[ib].scales[is+2] >> 4));
    y[l+64] = ggml_cuda_cast<dst_t>(dall * (x[ib].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[ib].scales[is+4] >> 4));
    y[l+96] = ggml_cuda_cast<dst_t>(dall * (x[ib].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[ib].scales[is+6] >> 4));
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q3_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q3_K * x = (const block_q3_K *) vx;

    const int64_t r = tid/4;
    const int64_t t = r/2;
    const int64_t is0 = r%2;
    const int64_t l0 = 16*is0 + 4*(tid%4);
    const int64_t n = t / 4;
    const int64_t j = t - 4*n;

    uint8_t m = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    int shift = 2*j;

    int8_t us = is <  4 ? (x[ib].scales[is-0] & 0xF) | (((x[ib].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[ib].scales[is-0] & 0xF) | (((x[ib].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[ib].scales[is-8] >>  4) | (((x[ib].scales[is+0] >> 4) & 3) << 4) :
                          (x[ib].scales[is-8] >>  4) | (((x[ib].scales[is-4] >> 6) & 3) << 4);
    float d_all = x[ib].d;
    float dl = d_all * (us - 32);

    dst_t * y = yy + 128*n + 32*j;
    const uint8_t * q = x[ib].qs + 32*n;
    const uint8_t * hm = x[ib].hmask;

    for (int l = l0; l < l0+4; ++l) {
        y[l] = ggml_cuda_cast<dst_t>(dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
    }
}

static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q4_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q4_K * x = (const block_q4_K *) vx;

    // assume 32 threads
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t is  = 2*il;
    const int64_t n   = 4;

    dst_t * y = yy + 64*il + n*ir;

    const float dall = __low2half(x[ib].dm);
    const float dmin = __high2half(x[ib].dm);

    const uint8_t * q = x[ib].qs + 32*il + n*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[ib].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[ib].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = ggml_cuda_cast<dst_t>(d1 * (q[l] & 0xF) - m1);
        y[l +32] = ggml_cuda_cast<dst_t>(d2 * (q[l] >>  4) - m2);
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q5_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q5_K * x = (const block_q5_K *) vx;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t il  = tid/16;   // il is in 0...3
    const int64_t ir  = tid%16;   // ir is in 0...15
    const int64_t is  = 2*il;     // is is in 0...6

    dst_t * y = yy + 64*il + 2*ir;

    const float dall = __low2half(x[ib].dm);
    const float dmin = __high2half(x[ib].dm);

    const uint8_t * ql = x[ib].qs + 32*il + 2*ir;
    const uint8_t * qh = x[ib].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[ib].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[ib].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t   hm  = 1 << (2*il);
    y[ 0] = ggml_cuda_cast<dst_t>(d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1);
    y[ 1] = ggml_cuda_cast<dst_t>(d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1);
    hm <<= 1;
    y[32] = ggml_cuda_cast<dst_t>(d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2);
    y[33] = ggml_cuda_cast<dst_t>(d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q6_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q6_K * x = (const block_q6_K *) vx;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t ip  = tid/32;   // ip is 0 or 1
    const int64_t il  = tid - 32*ip; // 0...32
    const int64_t is  = 8*ip + il/16;

    dst_t * y = yy + 128*ip + il;

    const float d = x[ib].d;

    const uint8_t * ql = x[ib].ql + 64*ip + il;
    const uint8_t   qh = x[ib].qh[32*ip + il];
    const int8_t  * sc = x[ib].scales + is;

    y[ 0] = ggml_cuda_cast<dst_t>(d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
    y[32] = ggml_cuda_cast<dst_t>(d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
    y[64] = ggml_cuda_cast<dst_t>(d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32));
    y[96] = ggml_cuda_cast<dst_t>(d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32));
}

//================================== i-quants

// Each call dequantizes one super-block of QK_K values into y with 32
// threads; iq4_nl packs QK_K/QK4_NL sub-blocks per super-block.

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq2_xxs(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq2_xxs * x = (const block_iq2_xxs  *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const uint16_t * q2 = x[ibs].qs + 4*ib;
    const uint8_t  * aux8 = (const uint8_t *)q2;
    const uint8_t  * grid = (const uint8_t *)(iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float)x[ibs].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 8; ++j) {
        y[j] = ggml_cuda_cast<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq2_xs(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq2_xs * x = (const block_iq2_xs *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const uint16_t * q2 = x[ibs].qs + 4*ib;
    const uint8_t  * grid = (const uint8_t *)(iq2xs_grid + (q2[il] & 511));
    const float d = (float)x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) {
        y[j] = ggml_cuda_cast<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq2_s(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq2_s * x = (const block_iq2_s *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const uint8_t * grid = (const uint8_t *)(iq2s_grid + (x[ibs].qs[4*ib+il] | ((x[ibs].qh[ib] << (8-2*il)) & 0x300)));
    const float d = (float)x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[ibs].qs[QK_K/8+4*ib+il];
    for (int j = 0; j < 8; ++j) {
        y[j] = ggml_cuda_cast<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq3_xxs(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq3_xxs * x = (const block_iq3_xxs  *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const uint8_t  * q3 = x[ibs].qs + 8*ib;
    const uint16_t * gas = (const uint16_t *)(x[ibs].qs + QK_K/4) + 2*ib;
    const uint8_t  * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*il+0]);
    const uint8_t  * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*il+1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float)x[ibs].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = ggml_cuda_cast<dst_t>(d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f));
        y[j+4] = ggml_cuda_cast<dst_t>(d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f));
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq3_s(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq3_s * x = (const block_iq3_s *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const uint8_t * qs = x[ibs].qs + 8*ib;
    const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*il+0] | ((x[ibs].qh[ib] << (8-2*il)) & 256)));
    const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*il+1] | ((x[ibs].qh[ib] << (7-2*il)) & 256)));
    const float d = (float)x[ibs].d * (1 + 2*((x[ibs].scales[ib/2] >> 4*(ib%2)) & 0xf));
    const uint8_t signs = x[ibs].signs[4*ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = ggml_cuda_cast<dst_t>(d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f));
        y[j+4] = ggml_cuda_cast<dst_t>(d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f));
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq1_s(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq1_s * x = (const block_iq1_s  *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const float delta = x[ibs].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d = (float)x[ibs].d * (2*((x[ibs].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4*ib+il] | (((x[ibs].qh[ib] >> 3*il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[j] = ggml_cuda_cast<dst_t>(d * (q[j] + delta));
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq1_m(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq1_m * x = (const block_iq1_m  *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const uint16_t * sc = (const uint16_t *)x[ibs].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2*ib + il/2; // sc[ib16/4] >> 3*(ib16%4) -> sc[ib/2] >> 3*((2*ib+il/2)%4);
    const float d = (float)scale.f16 * (2*((sc[ib16/4] >> 3*(ib16%4)) & 0x7) + 1);
    const float delta = x[ibs].qh[2*ib+il/2] & (0x08 << 4*(il%2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4*ib+il] | (((x[ibs].qh[2*ib+il/2] >> 4*(il%2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[j] = ggml_cuda_cast<dst_t>(d * (q[j] + delta));
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq4_nl(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq4_nl * x = (const block_iq4_nl *) vx + ibs*(QK_K/QK4_NL);

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 4*il;
    const uint8_t  * q4 = x[ib].qs + 4*il;
    const float d = (float)x[ib].d;
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = ggml_cuda_cast<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j+16] = ggml_cuda_cast<dst_t>(d * kvalues_iq4nl[q4[j] >>  4]);
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq4_xs(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {
    const block_iq4_xs * x = (const block_iq4_xs *)vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 4*il;
    const uint8_t  * q4 = x[ibs].qs + 16*ib + 4*il;
    const float d = (float)x[ibs].d * ((((x[ibs].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[ibs].scales_h >> 2*ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = ggml_cuda_cast<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j+16] = ggml_cuda_cast<dst_t>(d * kvalues_iq4nl[q4[j] >>  4]);
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_mxfp4(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_mxfp4 * x = (const block_mxfp4 *) vx + ibs*(QK_K/QK_MXFP4);

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 4*il;
    const uint8_t  * q4 = x[ib].qs + 4*il;
    const float d = ggml_cuda_e8m0_to_fp32(x[ib].e);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q4[j] & 0xf]*0.5f);
        y[j+16] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q4[j] >>  4]*0.5f);
    }
}
