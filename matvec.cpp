// matvec.cpp
// Usage:
//   g++ -O2 matvec_layout_unroll.cpp -o matvec
//   ./matvec <N> <M> <layout:contig|rows> [R=1] [unroll=1|4|8]
//
// Output CSV:
//   layout,N,M,R,unroll,seconds,GFLOPs,AI,ok
//
// Notes:
// - "rows" allocates each row separately (N allocations) -> more TLB/cache pressure.
// - "contig" uses one contiguous block (row-major).
// - Unrolling is manual (1/4/8). Compiler may still vectorize at -O2/-O3.
// - AI (arithmetic intensity) is the usual crude estimate for matvec.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <sys/time.h>

static inline double wall_seconds() {
    timeval tv{};
    gettimeofday(&tv, nullptr);
    return double(tv.tv_sec) + double(tv.tv_usec) * 1e-6;
}

// Matrix creation helpers
static inline void fill_uniform(std::vector<double>& v, uint64_t seed=42) {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> d(-0.5, 0.5);
    for (double &x : v) x = d(g);
}
static inline void fill_uniform_ptr(double* p, int64_t n, uint64_t seed=42) {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> d(-0.5, 0.5);
    for (int64_t i=0;i<n;++i) p[i] = d(g);
}

// arithmetic intensity (approx)
// AI ≈ (2*N*M) / (8*(N*M + M + N))  [flop/byte]  (double precision)
static inline double arithmetic_intensity(int64_t N, int64_t M) {
    long double fl = 2.0L*(long double)N*(long double)M;
    long double by = 8.0L*((long double)N*(long double)M +
                           (long double)M + (long double)N);
    return (double)(fl/by);
}

// correctness check (long double reference)
static inline bool validate_ref_contig(const std::vector<double>& A,
                                       const std::vector<double>& x,
                                       const std::vector<double>& y,
                                       int64_t N, int64_t M,
                                       double rtol=1e-10, double atol=1e-12)
{
    double max_err = 0.0, max_ref = 0.0;
    for (int64_t i=0;i<N;++i) {
        const double* Ai = &A[(size_t)i*(size_t)M];
        long double acc = 0.0L;
        for (int64_t j=0;j<M;++j) acc += (long double)Ai[j]*(long double)x[(size_t)j];
        double ref = (double)acc;
        double err = std::fabs(y[(size_t)i] - ref);
        if (err > max_err) max_err = err;
        double aref = std::fabs(ref);
        if (aref > max_ref) max_ref = aref;
    }
    return max_err <= (atol + rtol*max_ref);
}

static inline bool validate_ref_rows(double* const* Arows,
                                     const std::vector<double>& x,
                                     const std::vector<double>& y,
                                     int64_t N, int64_t M,
                                     double rtol=1e-10, double atol=1e-12)
{
    double max_err = 0.0, max_ref = 0.0;
    for (int64_t i=0;i<N;++i) {
        const double* Ai = Arows[i];
        long double acc = 0.0L;
        for (int64_t j=0;j<M;++j) acc += (long double)Ai[j]*(long double)x[(size_t)j];
        double ref = (double)acc;
        double err = std::fabs(y[(size_t)i] - ref);
        if (err > max_err) max_err = err;
        double aref = std::fabs(ref);
        if (aref > max_ref) max_ref = aref;
    }
    return max_err <= (atol + rtol*max_ref);
}

// matvec kernels (unrolled variants)
static inline void matvec_contig_u1(int64_t N, int64_t M,
                                    const std::vector<double>& A,
                                    const std::vector<double>& x,
                                    std::vector<double>& y)
{
    y.assign((size_t)N, 0.0);
    for (int64_t i=0;i<N;++i) {
        const double* Ai = &A[(size_t)i*(size_t)M];
        double acc=0.0;
        for (int64_t j=0;j<M;++j) acc += Ai[j]*x[(size_t)j];
        y[(size_t)i]=acc;
    }
}
static inline void matvec_contig_u4(int64_t N, int64_t M,
                                    const std::vector<double>& A,
                                    const std::vector<double>& x,
                                    std::vector<double>& y)
{
    y.assign((size_t)N, 0.0);
    for (int64_t i=0;i<N;++i) {
        const double* Ai = &A[(size_t)i*(size_t)M];
        double acc=0.0;
        int64_t j=0;
        for (; j+3<M; j+=4) {
            acc += Ai[j  ]*x[(size_t)j  ];
            acc += Ai[j+1]*x[(size_t)j+1];
            acc += Ai[j+2]*x[(size_t)j+2];
            acc += Ai[j+3]*x[(size_t)j+3];
        }
        for (; j<M; ++j) acc += Ai[j]*x[(size_t)j];
        y[(size_t)i]=acc;
    }
}
static inline void matvec_contig_u8(int64_t N, int64_t M,
                                    const std::vector<double>& A,
                                    const std::vector<double>& x,
                                    std::vector<double>& y)
{
    y.assign((size_t)N, 0.0);
    for (int64_t i=0;i<N;++i) {
        const double* Ai = &A[(size_t)i*(size_t)M];
        double acc=0.0;
        int64_t j=0;
        for (; j+7<M; j+=8) {
            acc += Ai[j  ]*x[(size_t)j  ];
            acc += Ai[j+1]*x[(size_t)j+1];
            acc += Ai[j+2]*x[(size_t)j+2];
            acc += Ai[j+3]*x[(size_t)j+3];
            acc += Ai[j+4]*x[(size_t)j+4];
            acc += Ai[j+5]*x[(size_t)j+5];
            acc += Ai[j+6]*x[(size_t)j+6];
            acc += Ai[j+7]*x[(size_t)j+7];
        }
        for (; j<M; ++j) acc += Ai[j]*x[(size_t)j];
        y[(size_t)i]=acc;
    }
}

static inline void matvec_rows_u1(int64_t N, int64_t M,
                                  double* const* Arows,
                                  const std::vector<double>& x,
                                  std::vector<double>& y)
{
    y.assign((size_t)N, 0.0);
    for (int64_t i=0;i<N;++i) {
        const double* Ai = Arows[i];
        double acc=0.0;
        for (int64_t j=0;j<M;++j) acc += Ai[j]*x[(size_t)j];
        y[(size_t)i]=acc;
    }
}
static inline void matvec_rows_u4(int64_t N, int64_t M,
                                  double* const* Arows,
                                  const std::vector<double>& x,
                                  std::vector<double>& y)
{
    y.assign((size_t)N, 0.0);
    for (int64_t i=0;i<N;++i) {
        const double* Ai = Arows[i];
        double acc=0.0;
        int64_t j=0;
        for (; j+3<M; j+=4) {
            acc += Ai[j  ]*x[(size_t)j  ];
            acc += Ai[j+1]*x[(size_t)j+1];
            acc += Ai[j+2]*x[(size_t)j+2];
            acc += Ai[j+3]*x[(size_t)j+3];
        }
        for (; j<M; ++j) acc += Ai[j]*x[(size_t)j];
        y[(size_t)i]=acc;
    }
}
static inline void matvec_rows_u8(int64_t N, int64_t M,
                                  double* const* Arows,
                                  const std::vector<double>& x,
                                  std::vector<double>& y)
{
    y.assign((size_t)N, 0.0);
    for (int64_t i=0;i<N;++i) {
        const double* Ai = Arows[i];
        double acc=0.0;
        int64_t j=0;
        for (; j+7<M; j+=8) {
            acc += Ai[j  ]*x[(size_t)j  ];
            acc += Ai[j+1]*x[(size_t)j+1];
            acc += Ai[j+2]*x[(size_t)j+2];
            acc += Ai[j+3]*x[(size_t)j+3];
            acc += Ai[j+4]*x[(size_t)j+4];
            acc += Ai[j+5]*x[(size_t)j+5];
            acc += Ai[j+6]*x[(size_t)j+6];
            acc += Ai[j+7]*x[(size_t)j+7];
        }
        for (; j<M; ++j) acc += Ai[j]*x[(size_t)j];
        y[(size_t)i]=acc;
    }
}

int main(int argc, char** argv) {
    if (argc < 4 || argc > 6) {
        std::fprintf(stderr, "Usage: %s <N> <M> <layout:contig|rows> [R=1] [unroll=1|4|8]\n", argv[0]);
        return 1;
    }
    const int64_t N = std::atoll(argv[1]);
    const int64_t M = std::atoll(argv[2]);
    const char* layout = argv[3];
    const int64_t R = (argc>=5) ? std::atoll(argv[4]) : 1;
    const int unroll = (argc==6) ? std::atoi(argv[5]) : 1;
    if (N<=0||M<=0||R<=0 || (unroll!=1 && unroll!=4 && unroll!=8)) {
        std::fprintf(stderr, "Bad arguments.\n"); return 1;
    }

    const double AI = arithmetic_intensity(N,M);

    std::vector<double> x((size_t)M), y((size_t)N, 0.0);
    fill_uniform(x, 1337);

    double t0=0.0, t1=0.0, elapsed=0.0;
    bool ok=false;

    if (std::strcmp(layout,"contig")==0) {
        // contiguous matrix
        std::vector<double> A((size_t)N*(size_t)M);
        fill_uniform(A, 42);

        // warmup
        switch(unroll){
            case 1: matvec_contig_u1(N,M,A,x,y); break;
            case 4: matvec_contig_u4(N,M,A,x,y); break;
            case 8: matvec_contig_u8(N,M,A,x,y); break;
        }

        t0 = wall_seconds();
        for (int64_t r=0;r<R;++r) {
            switch(unroll){
                case 1: matvec_contig_u1(N,M,A,x,y); break;
                case 4: matvec_contig_u4(N,M,A,x,y); break;
                case 8: matvec_contig_u8(N,M,A,x,y); break;
            }
        }
        t1 = wall_seconds();
        elapsed = t1 - t0;

        ok = validate_ref_contig(A, x, y, N, M);
    } else if (std::strcmp(layout,"rows")==0) {
        // one allocation per row
        std::vector<double*> Arows((size_t)N, nullptr);
        for (int64_t i=0;i<N;++i) {
            Arows[(size_t)i] = new (std::nothrow) double[(size_t)M];
            if (!Arows[(size_t)i]) { std::fprintf(stderr, "alloc fail\n"); return 2; }
            fill_uniform_ptr(Arows[(size_t)i], M, 42u + (uint64_t)i);
        }

        // warmup
        switch(unroll){
            case 1: matvec_rows_u1(N,M,Arows.data(),x,y); break;
            case 4: matvec_rows_u4(N,M,Arows.data(),x,y); break;
            case 8: matvec_rows_u8(N,M,Arows.data(),x,y); break;
        }

        t0 = wall_seconds();
        for (int64_t r=0;r<R;++r) {
            switch(unroll){
                case 1: matvec_rows_u1(N,M,Arows.data(),x,y); break;
                case 4: matvec_rows_u4(N,M,Arows.data(),x,y); break;
                case 8: matvec_rows_u8(N,M,Arows.data(),x,y); break;
            }
        }
        t1 = wall_seconds();
        elapsed = t1 - t0;

        ok = validate_ref_rows(Arows.data(), x, y, N, M);

        for (double* p : Arows) delete[] p;
    } else {
        std::fprintf(stderr, "layout must be 'contig' or 'rows'\n");
        return 1;
    }

    // FLOPs and performance
    const long double flops = (long double)R * 2.0L * (long double)N * (long double)M;
    const double gflops = (double)(flops / elapsed / 1e9L);

    // CSV
    std::printf("%s,%lld,%lld,%lld,%d,%.6f,%.3f,%.6f,%d\n",
        layout, (long long)N, (long long)M, (long long)R, unroll,
        elapsed, gflops, AI, ok ? 1 : 0);

    return ok ? 0 : 2;
}
