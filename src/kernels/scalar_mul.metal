// scalar_mul.metal — Metal compute kernel for scalar-vector multiply.
// NOTE: Metal compute shaders do not support double precision on Apple
// Silicon — uses float32 throughout. Caller MUST handle double<->float
// conversion at the R↔Metal boundary; expect ~1e-7 precision loss vs
// vDSP_vsmulD double path.

#include <metal_stdlib>
using namespace metal;

kernel void scalar_mul_kernel(
    device const float *v         [[buffer(0)]],
    device       float *out       [[buffer(1)]],
    constant     float &s         [[buffer(2)]],
    constant     uint  &n         [[buffer(3)]],
    uint id                       [[thread_position_in_grid]])
{
    if (id < n) {
        out[id] = s * v[id];
    }
}
