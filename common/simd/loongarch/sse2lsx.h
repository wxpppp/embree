#ifndef SSE2LSX_H
#define SSE2LSX_H

/* _mm_rcp_ps */
/*#ifndef SSE2LSX_PRECISE_DIV
#define SSE2LSX_PRECISE_DIV (0)
#endif*/
/* _mm_sqrt_ps and _mm_rsqrt_ps */
/*#ifndef SSE2LSX_PRECISE_SQRT
#define SSE2LSX_PRECISE_SQRT (0)
#endif*/
/* _mm_min_ps and _mm_max_ps */
/*#ifndef SSE2LSX_PRECISE_MINMAX
#define SSE2LSX_PRECISE_MINMAX (0)
#endif*/

/* compiler specific definitions */
#if defined(__GNUC__) || defined(__clang__)
#pragma push_macro("FORCE_INLINE")
#pragma push_macro("ALIGN_STRUCT")
#define FORCE_INLINE static inline __attribute__((always_inline))
#define ALIGN_STRUCT(x) __attribute__((aligned(x)))
#define _sse2lsx_likely(x) __builtin_expect(!!(x), 1)
#define _sse2lsx_unlikely(x) __builtin_expect(!!(x), 0)
#else
#pragma message("Macro name collisions may happen with unsupported compilers.")
#endif

/* C language does not allow initializing a variable with a function call. */
#ifdef __cplusplus
#define _sse2rvv_const static const
#else
#define _sse2rvv_const const
#endif

#define _MM_SHUFFLE(fp3, fp2, fp1, fp0) \
	(((fp3) << 6) | ((fp2) << 4) | ((fp1) << 2) | ((fp0)))

/* Rounding mode macros. */
#define _MM_FROUND_TO_NEAREST_INT 0x00
#define _MM_FROUND_TO_NEG_INF 0x01
#define _MM_FROUND_TO_POS_INF 0x02
#define _MM_FROUND_TO_ZERO 0x03
#define _MM_FROUND_CUR_DIRECTION 0x04
#define _MM_FROUND_NO_EXC 0x08
#define _MM_ROUND_NEAREST 0x0000
#define _MM_ROUND_DOWN 0x2000
#define _MM_ROUND_UP 0x4000
#define _MM_ROUND_TOWARD_ZERO 0x6000

#include <lsxintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

//typedef long long __m64 __attribute__ ((__vector_size__ (8), __may_alias__));

/* typedef long long __m128i __attribute__ ((__vector_size__ (16), __may_alias__));
 * 128-bit vector containing int64x2 or int32x4 
 */

/* typedef float __m128 __attribute__ ((__vector_size__ (16), __may_alias__));
 * 128-bit vector containing 4 floats
 */

/* typedef double _m128d __attribute__ ((__vector_size__ (16), __may_alias__));
 * 128-bit vector containing 2 doubles
 */

// A struct is defined in this header file called 'SIMDVec' which can be used
// by applications which attempt to access the contents of an __m128 struct
// directly.  It is important to note that accessing the __m128 struct directly
// is bad coding practice by Microsoft: @see:
// https://learn.microsoft.com/en-us/cpp/cpp/m128
//
// However, some legacy source code may try to access the contents of an __m128
// struct directly so the developer can use the SIMDVec as an alias for it.  Any
// casting must be done manually by the developer, as you cannot cast or
// otherwise alias the base NEON data type for intrinsic operations.
//
// union intended to allow direct access to an __m128 variable using the names
// that the MSVC compiler provides.  This union should really only be used when
// trying to access the members of the vector as integer values.  GCC/clang
// allow native access to the float members through a simple array access
// operator (in C since 4.6, in C++ since 4.8).
//
// Ideally direct accesses to SIMD vectors should not be used since it can cause
// a performance hit.  If it really is needed however, the original __m128
// variable can be aliased with a pointer to this union and used to access
// individual components.  The use of this union should be hidden behind a macro
// that is used throughout the codebase to access the members instead of always
// declaring this type of variable.
typedef union ALIGN_STRUCT(16) SIMDVec {
  float m128_f32[4];    // as floats - DON'T USE. Added for convenience.
  int8_t m128_i8[16];   // as signed 8-bit integers.
  int16_t m128_i16[8];  // as signed 16-bit integers.
  int32_t m128_i32[4];  // as signed 32-bit integers.
  int64_t m128_i64[2];  // as signed 64-bit integers.
  uint8_t m128_u8[16];  // as unsigned 8-bit integers.
  uint16_t m128_u16[8]; // as unsigned 16-bit integers.
  uint32_t m128_u32[4]; // as unsigned 32-bit integers.
  uint64_t m128_u64[2]; // as unsigned 64-bit integers.
} SIMDVec;

// __int64 is defined in the Intrinsics Guide which maps to different datatype
// in different data model
#if !(defined(_WIN32) || defined(_WIN64) || defined(__int64))
#if (defined(__x86_64__) || defined(__i386__))
#define __int64 long long
#elif defined(__loongarch__)
#define __int64 long long
#else
#define __int64 int64_t
#endif
#endif
 

//pass
FORCE_INLINE __m128i _mm_set_epi32 (int d, int c, int b, int a) {
  v4i32 result = {a, b, c, d};
  return (__m128i)result;
}

//pass
FORCE_INLINE __m128i _mm_set_epi64x( __int64 b, __int64 a) {
  //v2i64 result = {a, b};
  //return (__m128i)result;

  return __lsx_vinsgr2vr_d(__lsx_vreplgr2vr_d(b), a, 0);
}

//pass
FORCE_INLINE __m128 _mm_set1_ps(float a) {
// fail  return (__m128)__lsx_vreplgr2vr_w(a);
  v4f32 result = {a, a, a, a};
  return (__m128)result;
}

//pass
FORCE_INLINE __m128 _mm_rcp_ps(__m128 a) {
/*
  float buffer[4];
  for (int i = 0; i < 4; i++) {
    buffer[i] = 1.0f / ((float*)&a)[i];
  }
  v4f32 result = {buffer[0], buffer[1], buffer[2], buffer[3]};
  return (__m128)result;
*/  
  return __lsx_vfrecip_s(a);
}

//pass
FORCE_INLINE float _mm_cvtss_f32(__m128 a) {
  return ((float*)&a)[0];
}

//pass
FORCE_INLINE int _mm_cvtsi128_si32 (__m128i a) {
  return __lsx_vpickve2gr_w(a, 0);
}

//pass
FORCE_INLINE __int64 _mm_cvtsi128_si64 (__m128i a) {
  return __lsx_vpickve2gr_d(a, 0);
}

//pass
FORCE_INLINE __m128 _mm_div_ps(__m128 a, __m128 b) {
  return __lsx_vfdiv_s(a, b);
}

//pass
FORCE_INLINE __m128 _mm_sqrt_ps(__m128 a) {
  return __lsx_vfsqrt_s(a);
}

//pass
FORCE_INLINE __m128 _mm_rsqrt_ps(__m128 a) {
  return __lsx_vfrsqrt_s(a);
}

//pass
FORCE_INLINE __m128i _mm_add_epi32(__m128i a, __m128i b) {
  return __lsx_vadd_w(a, b);
}

//pass
FORCE_INLINE __m128 _mm_add_ps(__m128 a, __m128 b) {
  return  __lsx_vfadd_s(a, b);
}

//pass
FORCE_INLINE __m128i _mm_sub_epi32(__m128i a, __m128i b) {
  return __lsx_vsub_w(a, b);
}

//pass
FORCE_INLINE __m128 _mm_sub_ps(__m128 a, __m128 b) {
  return __lsx_vfsub_s(a, b);
}

//pass
FORCE_INLINE __m128 _mm_mul_ps(__m128 a, __m128 b) {
  return __lsx_vfmul_s(a, b);
}

//pass
FORCE_INLINE __m128 _mm_cvtepi32_ps(__m128i a) {
  return __lsx_vffint_s_w(a);
}

//pass
FORCE_INLINE __m128 _mm_set_ps(float d, float c, float b, float a) {
  v4f32 result = {a, b, c, d};
  return (__m128)result;
}

//pass
FORCE_INLINE __m128i _mm_set1_epi32(int a) {
  return __lsx_vreplgr2vr_w(a);
}

//pass
FORCE_INLINE __m128i _mm_cvtps_epi32(__m128 a) {
  return __lsx_vftint_w_s(a);
}

//pass
FORCE_INLINE __m128 _mm_undefined_ps(void) {
  return (__m128)__lsx_vldi(0);
}

//pass
FORCE_INLINE __m128i _mm_load_si128(__m128i const *mem_addr) {
  return __lsx_vld(mem_addr, 0);
}

//pass
FORCE_INLINE __m128 _mm_load_ps(float const *mem_addr) {
  return (__m128)__lsx_vld(mem_addr, 0);
}

//pass
FORCE_INLINE __m128 _mm_loadu_ps(float const *mem_addr) {
  return (__m128)__lsx_vld(mem_addr, 0);
}

//pass
FORCE_INLINE __m128i _mm_loadu_si128(__m128i const *mem_addr) {
  return __lsx_vld(mem_addr, 0);
}

//pass
FORCE_INLINE void _mm_store_ps(float *mem_addr, __m128 a) {
  return __lsx_vst(a, mem_addr, 0);
}

//pass
FORCE_INLINE void _mm_storeu_ps(float *mem_addr, __m128 a) {
  return __lsx_vst(a, mem_addr, 0);
}

//pass
FORCE_INLINE void _mm_storeu_pd(double *mem_addr, __m128d a) {
  return __lsx_vst(a, mem_addr, 0);
}

//pass
FORCE_INLINE void _mm_store_si128(__m128i *mem_addr, __m128i a) {
  return __lsx_vst(a, mem_addr, 0);
}

//pass
FORCE_INLINE void _mm_storeu_si128(__m128i *mem_addr, __m128i a) {
  return __lsx_vst(a, mem_addr, 0);
}

//pass
FORCE_INLINE __m128 _mm_castsi128_ps(__m128i a) {
  return (__m128)a;
}

//pass
FORCE_INLINE __m128d _mm_castsi128_pd(__m128i a) {
  return (__m128d)a;
}

//pass
FORCE_INLINE __m128 _mm_min_ps(__m128 a, __m128 b) {
  return __lsx_vfmin_s(a, b);
}

// FORCE_INLINE __m128i _mm_min_epi32(__m128i a, __m128i b) { }

// FORCE_INLINE __m128i _mm_min_epu32(__m128i a, __m128i b) { }

// FORCE_INLINE __m128i _mm_max_epu32(__m128i a, __m128i b) { }

//pass
FORCE_INLINE __m128 _mm_max_ps(__m128 a, __m128 b) {
  return __lsx_vfmax_s(a, b);
}

// FORCE_INLINE __m128i _mm_max_epi32(__m128i a, __m128i b) { }

//pass
FORCE_INLINE __m128 _mm_xor_ps(__m128 a, __m128 b) {
  return (__m128)__lsx_vxor_v((__m128i)a, (__m128i)b);
}

//pass
FORCE_INLINE __m128i _mm_xor_si128(__m128i a, __m128i b) {
  return __lsx_vxor_v(a, b);
}

#define _mm_slli_epi32(a, b) __lsx_vslli_w((__m128i)(a), (b))
#define _mm_srai_epi32(a, b) __lsx_vsrai_w((__m128i)(a), (b))
//#define _mm_srli_epi32(a, b) __lsx_vsrli_w((__m128i)(a), (b))
#define _mm_srli_si128(a, b) __lsx_vbsrl_v((a), (b))

/*FORCE_INLINE __m128i _mm_slli_epi32(__m128i a, int imm8) {
  const unsigned char _imm8 = imm8 & 0xff;
  return __lsx_vslli_w(a, _imm8);
}*/

/*FORCE_INLINE __m128i _mm_srai_epi32(__m128i a, int imm8) {
  return __lsx_vsrai_w(a, imm8);
}*/

FORCE_INLINE __m128i _mm_srli_epi32(__m128i a, int b) {
   int arr[4];
   _mm_storeu_si128((__m128i*)arr, a);

   for (int i = 0; i < 4; i++) {
        arr[i] >>= b;
   }

   return _mm_loadu_si128((__m128i*)arr);
}

/*FORCE_INLINE __m128i _mm_srli_si128(__m128i a, int imm8) {
  return __lsx_vbsrl_v(a, imm8);
}*/


//pass
FORCE_INLINE __m128 _mm_or_ps(__m128 a, __m128 b) {
  return (__m128)__lsx_vor_v((__m128i)a, (__m128i)b);
}

//pass
FORCE_INLINE __m128i _mm_or_si128(__m128i a, __m128i b) {
  return __lsx_vor_v(a, b);
}

//pass
FORCE_INLINE __m128 _mm_and_ps(__m128 a, __m128 b) {
  return (__m128)__lsx_vand_v((__m128i)a, (__m128i)b);
}

//pass
FORCE_INLINE __m128i _mm_and_si128(__m128i a, __m128i b) {
  return __lsx_vand_v(a, b);
}

//pass
FORCE_INLINE __m128 _mm_andnot_ps(__m128 a, __m128 b) {
  return (__m128)__lsx_vandn_v((__m128i)a, (__m128i)b);
}

//pass
FORCE_INLINE __m128i _mm_andnot_si128(__m128i a, __m128i b) {
  return __lsx_vandn_v(a, b);
}

//pass
FORCE_INLINE __m128i _mm_castps_si128(__m128 a) {
  return (__m128i)a;
}

//pass
FORCE_INLINE __m128d _mm_castps_pd(__m128 a) {
  return (__m128d)a;
}

//pass
FORCE_INLINE __m128 _mm_setzero_ps(void) {
  return (__m128)__lsx_vldi(0);
}

//pass
FORCE_INLINE __m128i _mm_setzero_si128() {
  return __lsx_vldi(0);
}

//pass
FORCE_INLINE __m128i _mm_cmpeq_epi32(__m128i a, __m128i b) {
  return __lsx_vseq_w(a, b);
}

//pass
FORCE_INLINE __m128 _mm_cmpeq_ps(__m128 a, __m128 b) {
  return (__m128)__lsx_vfcmp_ceq_s(a, b);
}

//pass
FORCE_INLINE __m128 _mm_cmpneq_ps(__m128 a, __m128 b) {
  return (__m128)__lsx_vfcmp_cune_s(a, b);
}

//pass
FORCE_INLINE __m128 _mm_cmpnlt_ps(__m128 a, __m128 b) {
  return (__m128)__lsx_vfcmp_cule_s(b, a);
}

//pass
FORCE_INLINE __m128 _mm_cmpnle_ps(__m128 a, __m128 b) {
  return (__m128)__lsx_vfcmp_cult_s(b, a);
}

//pass
FORCE_INLINE __m128 _mm_cmple_ps(__m128 a, __m128 b) {
  return (__m128)__lsx_vfcmp_cle_s(a, b);
}

//pass
FORCE_INLINE __m128i _mm_cmplt_epi32(__m128i a, __m128i b) {
  return __lsx_vslt_w(a, b);
}

//pass
FORCE_INLINE __m128 _mm_cmplt_ps(__m128 a, __m128 b) {
  return (__m128)__lsx_vfcmp_clt_s(a, b);
}

//pass
FORCE_INLINE __m128i _mm_cmpgt_epi32(__m128i a, __m128i b) {
  return __lsx_vslt_w(b, a);
}

//pass
FORCE_INLINE int _mm_movemask_ps(__m128 a) {
  return __lsx_vpickve2gr_wu(__lsx_vmskltz_w((__m128i)a), 0);
}

//pass
FORCE_INLINE __m128 _mm_unpacklo_ps(__m128 a, __m128 b) {
  return (__m128)__lsx_vilvl_w((__m128i)b, (__m128i)a);
}

//pass
FORCE_INLINE __m128 _mm_unpackhi_ps(__m128 a, __m128 b) {
  return (__m128)__lsx_vilvh_w((__m128i)b, (__m128i)a);
}

#define _mm_shuffle_epi32(a, b) __lsx_vshuf4i_w((__m128i)(a), (b))
#define _mm_shuffle_ps(a, b, c) (__m128)__lsx_vpermi_w((__m128)(b), (__m128)(a), (c))

/*FORCE_INLINE __m128i _mm_shuffle_epi32(__m128i a, int imm8) {
  return __lsx_vshuf4i_w(a, imm8);
}*/

/*FORCE_INLINE __m128 _mm_shuffle_ps(__m128 a, __m128 b, unsigned int imm8) {
  return (__m128)__lsx_vpermi_w(b, a, imm8);
}*/

FORCE_INLINE void _mm_pause() {
//  __asm__ __volatile__("dbar 0" ::: "memory");
  __asm__ __volatile__("ibar 0" ::: "memory");
}

/*FORCE_INLINE __m128 _mm_move_ss(__m128 a, __m128 b) {

}*/

enum _mm_hint {
  _MM_HINT_NTA = 0,
  _MM_HINT_T0 = 1,
  _MM_HINT_T1 = 2,
  _MM_HINT_T2 = 3,
};

FORCE_INLINE void _mm_prefetch(const void *p, int i) {
  (void) i;
  __builtin_prefetch(p);
}

FORCE_INLINE __m128 _mm_set_ss(float a) {
  v4f32 result = {a, 0.0f, 0.0f, 0.0f};
  return (__m128)result;
}

FORCE_INLINE __m128 _mm_rcp_ss(__m128 a) {
//  float lowest = _mm_cvtss_f32(_mm_rcp_ps(a));
  v4f32 buf = _mm_load_ps((float*)&a);
  v4f32 result = {1.0f / buf[0], buf[1], buf[2], buf[3]};
  return (__m128)result;
}

FORCE_INLINE __m128 _mm_mul_ss(__m128 a, __m128 b) {
  v4f32 buf1 = _mm_load_ps((float*)&a);
  v4f32 buf2 = _mm_load_ps((float*)&b);
  buf1[0] = buf1[0] * buf2[0];
  return (__m128)buf1;
}

FORCE_INLINE __m128 _mm_rsqrt_ss(__m128 in) {
  v4f32 buf = _mm_load_ps((float*)&in);
  v4f32 buf1 = (v4f32)_mm_rsqrt_ps(in);
  v4f32 result = {buf1[0], buf[1], buf[2], buf[3]};
  return (__m128)result;
}

FORCE_INLINE __m128 _mm_sub_ss(__m128 a, __m128 b) {
  v4f32 buf1 = _mm_load_ps((float*)&a);
  v4f32 buf2 = _mm_load_ps((float*)&b);
  v4f32 result = {buf1[0] - buf2[0], buf1[1], buf1[2], buf1[3]};
  return (__m128)result;
}

FORCE_INLINE __m128 _mm_add_ss(__m128 a, __m128 b) {
  v4f32 buf1 = _mm_load_ps((float*)&a);
  v4f32 buf2 = _mm_load_ps((float*)&b);
  v4f32 result = {buf1[0] + buf2[0], buf1[1], buf1[2], buf1[3]};
  return (__m128)result;
}

FORCE_INLINE void *_mm_malloc(size_t size, size_t align) {
  void *ptr;
  if (align == 1) {
    return malloc(size);
  }
  if (align == 2 || (sizeof(void *) == 8 && align == 4)) {
    align = sizeof(void *);
  }
  ptr = aligned_alloc(align, size);
  if (ptr) {
    return ptr;
  }
  return NULL;
}

FORCE_INLINE void _mm_free(void *mem_addr) { free(mem_addr); }

// The bit field mapping to the FCSR (floating-point control and status
// register)

typedef struct {
  uint8_t bit0_7;
  uint8_t bit_8 : 1;
  uint8_t bit_9 : 1;
  uint32_t bit10_31 : 22;
} fcsr_bitfield;

FORCE_INLINE void _MM_SET_ROUNDING_MODE(int rounding) {
  union {
        fcsr_bitfield field;
        uint32_t value;
  } r;

  __asm__ volatile("movfcsr2gr %0, $fcsr3" : "=r"(r.value));
  switch (rounding) {
    case _MM_ROUND_TOWARD_ZERO:
        r.field.bit_8 = 1;
        r.field.bit_9 = 0;
        break;
    case _MM_ROUND_DOWN:
        r.field.bit_8 = 1;
        r.field.bit_9 = 1;
        break;
    case _MM_ROUND_UP:
        r.field.bit_8 = 0;
        r.field.bit_9 = 1;
        break;
    default:  //_MM_ROUND_NEAREST
        r.field.bit_8 = 0;
        r.field.bit_9 = 0;
    }
  __asm__ volatile("movgr2fcsr $fcsr3, %0" : "=r"(r));
}

FORCE_INLINE void _mm_setcsr(unsigned int a) {
   _MM_SET_ROUNDING_MODE(a);
}


FORCE_INLINE int _mm_getcsr() {
  return 0;
}

FORCE_INLINE void _mm_mfence(void) {
  __sync_synchronize();
}

/* Dummy defines for floating point control */
#define _MM_MASK_MASK 0x1f80
#define _MM_MASK_DIV_ZERO 0x200
#define _MM_FLUSH_ZERO_ON 0x8000
#define _MM_MASK_DENORM 0x100
#define _MM_SET_EXCEPTION_MASK(x)
#define _MM_SET_FLUSH_ZERO_MODE(x)

#endif
