# Matrix–Vector Multiplication: Roofline Report

## Introduction
We benchmarked dense matrix–vector multiplication (matvec) with different memory layouts, loop unrolling factors, and compiler optimizations. Performance was analyzed with the roofline model on peak BW = 160 GB/s, peak compute ≈ 1 TFLOP/s. Arithmetic intensity (AI) is ≈ 0.25 FLOP/Byte, making matvec strongly memory-bound.

## Variants
- **Memory allocation**
  - *contig*: contiguous row-major block
  - *rows*: separate allocation per row  
- **Loop unrolling**
  - u1, u4, u8  
- **Compiler optimization**
  - `-O0`, `-O2`, `-O3`  


## Results
### `-O0`
- Very low performance (~1–2 GFLOP/s).
- No benefit from unrolling or layout.
- Far below memory roofline.

### `-O2`
- Performance improved (~2–4 GFLOP/s).
- Contiguous layout > rows (better cache use).
- Small gains from manual unrolling.

### `-O3`
- Best results (~5–6 GFLOP/s).
- Contig + u8 highest performance.
- Row layout slower due to pointer chasing/TLB misses.
- Still below predicted roofline (~51 GFLOP/s).


## Analysis
- **Memory-bound behavior**: AI too low to exploit peak compute, bandwidth dominates.
- **Compiler optimizations**: Crucial for performance; O0 vs O3 differed by ~5×.
- **Layout**: Contiguous allocation consistently faster.
- **Unrolling**: Small gains, overshadowed by compiler auto-unrolling.


## Conclusion
Matvec remains bandwidth-limited, with performance far below the theoretical roofline.  
- Optimizations (`-O3`) and contiguous allocation are essential.  
- Unrolling provides minor improvements.  
- The experiment highlights the gap between peak hardware limits and practical performance.  
