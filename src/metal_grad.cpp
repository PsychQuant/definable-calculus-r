// metal_grad.cpp — Objective-C++ bridge for the Metal compute backend.
//
// add-metal-backend (2026-05-29): promoted from the inst/metal/ architectural
// artifact into the active build (Path A). Compiled as Objective-C++ via the
// global `-x objective-c++ -fobjc-arc` flag in src/Makevars (Obj-C++ is a C++
// superset, so the package's other .cpp files compile unchanged); linked with
// `-framework Metal -framework Foundation` alongside `-framework Accelerate`.
// Using `.cpp` (not `.mm`) keeps Rcpp::compileAttributes() scanning the
// [[Rcpp::export]] tags, and the global flag avoids the per-file Makevars rule
// that previously broke $(SHLIB) chaining.
//
// Dispatches a float32 scalar-vector multiply to a Metal compute kernel
// (scalar_mul_kernel in inst/metal/scalar_mul.metallib). Only worthwhile at
// very large n (the per-call double<->float32 conversion passes mean Metal
// does not beat vDSP until n is enormous); the R-side threshold guard
// (.metal_path_available + dat.metal_threshold) decides when to route here.

#include <Rcpp.h>
#include <vector>
#include <string>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

static id<MTLDevice> g_device = nil;
static id<MTLLibrary> g_library = nil;
static id<MTLComputePipelineState> g_pipeline = nil;
static id<MTLCommandQueue> g_queue = nil;

static bool metal_init(const char *metallib_path) {
  if (g_pipeline) return true;
  g_device = MTLCreateSystemDefaultDevice();
  if (!g_device) return false;
  NSError *err = nil;
  NSString *path = [NSString stringWithUTF8String:metallib_path];
  NSURL *url = [NSURL fileURLWithPath:path];
  g_library = [g_device newLibraryWithURL:url error:&err];
  if (!g_library) return false;
  id<MTLFunction> fn = [g_library newFunctionWithName:@"scalar_mul_kernel"];
  if (!fn) return false;
  g_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&err];
  if (!g_pipeline) return false;
  g_queue = [g_device newCommandQueue];
  return g_queue != nil;
}

//' Initialize the Metal scalar-multiply pipeline (internal)
//'
//' Loads the pre-compiled metallib and builds the compute pipeline. Returns
//' FALSE (never raises) when no Metal device / library / kernel is available,
//' so the R-side guard can fall back to the CPU path. Idempotent: a second
//' call with an already-built pipeline returns TRUE immediately.
//'
//' @param metallib_path Filesystem path to scalar_mul.metallib.
//' @return TRUE if the pipeline is ready, FALSE otherwise.
//' @export
// [[Rcpp::export]]
bool metal_scalar_mul_init(std::string metallib_path) {
  @autoreleasepool { return metal_init(metallib_path.c_str()); }
}

//' Scalar-vector multiply on the Metal GPU (internal)
//'
//' Computes \code{s * v} elementwise via the Metal compute kernel. Inputs are
//' converted to float32 for the GPU and back to double on return, so the
//' result matches the vDSP/base-R double product to within ~1e-6 relative.
//' Requires \code{metal_scalar_mul_init} to have succeeded.
//'
//' @param s Scalar multiplier.
//' @param v Numeric vector.
//' @return Numeric vector equal to s * v (float32 precision).
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector metal_scalar_mul(double s, Rcpp::NumericVector v) {
  @autoreleasepool {
    if (!g_pipeline) Rcpp::stop("metal_scalar_mul: pipeline not initialized");
    R_xlen_t n = v.size();
    Rcpp::NumericVector out(Rcpp::no_init(n));
    if (n == 0) return out;
    std::vector<float> v_f(n), out_f(n);
    for (R_xlen_t i = 0; i < n; ++i) v_f[i] = static_cast<float>(v[i]);
    float s_f = static_cast<float>(s);
    uint32_t n_u = static_cast<uint32_t>(n);
    id<MTLBuffer> v_buf = [g_device newBufferWithBytes:v_f.data()
                                                length:n * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_buf = [g_device newBufferWithLength:n * sizeof(float)
                                                  options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_pipeline];
    [enc setBuffer:v_buf offset:0 atIndex:0];
    [enc setBuffer:out_buf offset:0 atIndex:1];
    [enc setBytes:&s_f length:sizeof(float) atIndex:2];
    [enc setBytes:&n_u length:sizeof(uint32_t) atIndex:3];
    NSUInteger threads = g_pipeline.maxTotalThreadsPerThreadgroup;
    if (threads > n_u) threads = n_u;
    [enc dispatchThreads:MTLSizeMake(n_u, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    float *result = (float *)[out_buf contents];
    for (R_xlen_t i = 0; i < n; ++i) out[i] = static_cast<double>(result[i]);
    return out;
  }
}
