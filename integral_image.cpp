/**
 * integral_image.cpp
 * ==================
 * Integral Image (Summed Area Table) — C++17 con OpenMP
 *
 * Compile:
 *   g++ -std=c++17 -O2 -march=native -fopenmp -o integral_image integral_image.cpp
 *
 * Usage:
 *   ./integral_image [width] [height] [max_threads] [num_runs]
 *
 * Output:
 *   results_scaling.csv   (speedup vs thread count)
 *   results_size.csv      (performance vs image size)
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>


using u8   = uint8_t;
using i64  = int64_t;
using f64  = double;

// ─── Timer ────────────────────────────────────────────────────────────────────

class Timer {
    using Clock = std::chrono::steady_clock;
    Clock::time_point t0_;
public:
    Timer() : t0_(Clock::now()) {}
    f64 elapsed() const {
        return std::chrono::duration<f64>(Clock::now() - t0_).count();
    }
    void reset() { t0_ = Clock::now(); }
};

// ─── Image ────────────────────────────────────────────────────────────────────

/**
 * Immagine grayscale sintetica generata con una combinazione di
 * funzioni sinusoidali + rumore uniforme per risultati riproducibili.
 */
std::vector<u8> make_image(int W, int H, unsigned seed = 42)
{
    std::vector<u8> img(static_cast<size_t>(W) * H);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> noise(-15, 15);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            double v = 128.0
                + 60.0 * std::sin(2.0 * M_PI * x / W)
                + 40.0 * std::cos(2.0 * M_PI * y / H)
                + noise(rng);
            v = std::clamp(v, 0.0, 255.0);
            img[y * W + x] = static_cast<u8>(v);
        }
    }
    return img;
}

// ─── Integral Image — versione sequenziale ────────────────────────────────────

/**
 * Calcola la Summed Area Table con la ricorrenza standard:
 *
 *   II(x,y) = img(x,y) + II(x-1,y) + II(x,y-1) - II(x-1,y-1)
 *
 * Complessità: O(W·H) tempo, O(W·H) spazio.
 * Tipo i64 per evitare overflow su immagini di grandi dimensioni.
 */
std::vector<i64> integral_sequential(const std::vector<u8>& img, int W, int H)
{
    std::vector<i64> II(static_cast<size_t>(W) * H, 0);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            i64 v = img[y * W + x];
            if (x > 0) v += II[y * W + (x - 1)];
            if (y > 0) v += II[(y - 1) * W + x];
            if (x > 0 && y > 0) v -= II[(y - 1) * W + (x - 1)];
            II[y * W + x] = v;
        }
    }
    return II;
}

// ─── Integral Image — versione parallela (OpenMP, two-pass) ──────────────────

/**
 * Algoritmo two-pass parallelizzabile:
 *
 *   Pass 1: prefix sum per RIGHE  → ogni riga è indipendente → parallelizzabile
 *   Pass 2: prefix sum per COLONNE → ogni colonna è indipendente → parallelizzabile
 *
 * Dopo i due passaggi, II contiene la SAT completa, identica alla versione
 * sequenziale. schedule(static) garantisce bilanciamento ottimale su righe/
 * colonne di uguale lunghezza.
 */
std::vector<i64> integral_parallel(const std::vector<u8>& img,
                                   int W, int H, int nthreads)
{
    std::vector<i64> II(static_cast<size_t>(W) * H, 0);

    // Copy + cast: ogni thread scrive su porzioni disgiunte
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x)
            II[y * W + x] = static_cast<i64>(img[y * W + x]);
    }

    // Pass 1 — row-wise prefix sum (parallelismo perfetto: zero dipendenze tra righe)
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int y = 0; y < H; ++y) {
        i64* row = II.data() + y * W;
        for (int x = 1; x < W; ++x)
            row[x] += row[x - 1];
    }

    // Pass 2 — column-wise prefix sum (parallelismo perfetto: zero dipendenze tra colonne)
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int x = 0; x < W; ++x) {
        for (int y = 1; y < H; ++y)
            II[y * W + x] += II[(y - 1) * W + x];
    }

    return II;
}

// ─── Query O(1) ───────────────────────────────────────────────────────────────

/**
 * Calcola la somma di una regione rettangolare [x1,y1] → [x2,y2]
 * in O(1) usando i 4 angoli della SAT (inclusione-esclusione).
 */
inline i64 region_sum(const std::vector<i64>& II, int W,
                      int x1, int y1, int x2, int y2) noexcept
{
    i64 s = II[y2 * W + x2];
    if (x1 > 0) s -= II[y2 * W + (x1 - 1)];
    if (y1 > 0) s -= II[(y1 - 1) * W + x2];
    if (x1 > 0 && y1 > 0) s += II[(y1 - 1) * W + (x1 - 1)];
    return s;
}

// ─── Verifica correttezza ─────────────────────────────────────────────────────

bool verify(const std::vector<i64>& ref, const std::vector<i64>& par)
{
    return ref == par;   // element-wise comparison via std::vector::operator==
}

// ─── Struttura risultato benchmark ───────────────────────────────────────────

struct BenchResult {
    int  threads;
    f64  time_seq_ms;
    f64  time_par_ms;
    f64  speedup;
    f64  efficiency;
    bool ok;
};

// ─── Benchmark ────────────────────────────────────────────────────────────────

BenchResult run_benchmark(int W, int H, int nthreads, int nruns)
{
    auto img = make_image(W, H);

    // Sequenziale
    f64 t_seq = 0.0;
    std::vector<i64> II_ref;
    for (int r = 0; r < nruns; ++r) {
        Timer t;
        auto tmp = integral_sequential(img, W, H);
        t_seq += t.elapsed();
        if (r == 0) II_ref = std::move(tmp);
    }
    t_seq = t_seq / nruns * 1e3; // → ms

    // Parallela
    f64 t_par = 0.0;
    std::vector<i64> II_par;
    for (int r = 0; r < nruns; ++r) {
        Timer t;
        auto tmp = integral_parallel(img, W, H, nthreads);
        t_par += t.elapsed();
        if (r == 0) II_par = std::move(tmp);
    }
    t_par = t_par / nruns * 1e3;

    bool ok = verify(II_ref, II_par);
    f64 speedup = t_seq / t_par;

    return { nthreads, t_seq, t_par, speedup, speedup / nthreads, ok };
}

// ─── Demo query O(1) ─────────────────────────────────────────────────────────



// ─── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[])
{
    int W_def       = (argc > 1) ? std::stoi(argv[1]) : 4096;
    int H_def       = (argc > 2) ? std::stoi(argv[2]) : 4096;
    int max_threads = (argc > 3) ? std::stoi(argv[3]) : omp_get_max_threads();
    int nruns       = (argc > 4) ? std::stoi(argv[4]) : 5;

    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║    Integral Image Benchmark  —  C++17 + OpenMP          ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Immagine : " << W_def << "×" << H_def << " pixel\n";
    std::cout << "║  Thread   : 1.." << max_threads << "\n";
    std::cout << "║  Run/conf : " << nruns << "\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";

    // ── 1. Thread scaling ──────────────────────────────────────────────────────
    std::cout << "┌──────────┬──────────┬──────────┬──────────┬────────────┐\n";
    std::cout << "│ Thread   │ Seq (ms) │ Par (ms) │ Speedup  │ Efficienza │\n";
    std::cout << "├──────────┼──────────┼──────────┼──────────┼────────────┤\n";

    std::ofstream csv_scale("results_scaling.csv");
    csv_scale << "threads,seq_ms,par_ms,speedup,efficiency\n";

    for (int t = 1; t <= max_threads; ++t) {
        auto r = run_benchmark(W_def, H_def, t, nruns);
        if (!r.ok) { std::cerr << "ERRORE: verifica fallita thread=" << t << "\n"; return 1; }
        std::cout << "│ " << std::setw(8) << t
                  << " │ " << std::setw(8) << std::fixed << std::setprecision(2) << r.time_seq_ms
                  << " │ " << std::setw(8) << r.time_par_ms
                  << " │ " << std::setw(8) << std::setprecision(3) << r.speedup
                  << " │ " << std::setw(7)  << std::setprecision(1) << r.efficiency*100 << "% │\n";
        csv_scale << t << "," << r.time_seq_ms << "," << r.time_par_ms
                  << "," << r.speedup << "," << r.efficiency << "\n";
    }
    std::cout << "└──────────┴──────────┴──────────┴──────────┴────────────┘\n\n";
    csv_scale.close();

    // ── 2. Size scaling ───────────────────────────────────────────────────────
    std::cout << "┌──────────────┬──────────┬──────────┬──────────┬──────────┐\n";
    std::cout << "│ Pixel totali │ Seq (ms) │ Par (ms) │ Speedup  │ MP/s par │\n";
    std::cout << "├──────────────┼──────────┼──────────┼──────────┼──────────┤\n";

    std::ofstream csv_size("results_size.csv");
    csv_size << "size,seq_ms,par_ms,speedup,mpps\n";

    const int SIZES[] = {256, 512, 1024, 2048, 4096, 8192};
    for (int S : SIZES) {
        auto r = run_benchmark(S, S, max_threads, nruns);
        f64 mpps = static_cast<f64>(S) * S / (r.time_par_ms / 1e3) / 1e6;
        std::cout << "│ " << std::setw(12) << i64(S)*S
                  << " │ " << std::setw(8) << std::fixed << std::setprecision(2) << r.time_seq_ms
                  << " │ " << std::setw(8) << r.time_par_ms
                  << " │ " << std::setw(8) << std::setprecision(3) << r.speedup
                  << " │ " << std::setw(7)  << std::setprecision(1) << mpps << " │\n";
        csv_size << i64(S)*S << "," << r.time_seq_ms << "," << r.time_par_ms
                 << "," << r.speedup << "," << mpps << "\n";
    }
    std::cout << "└──────────────┴──────────┴──────────┴──────────┴──────────┘\n\n";
    csv_size.close();

    demo_queries(1024, 1024);

    std::cout << "CSV salvati: results_scaling.csv  results_size.csv\n";
    std::cout << "Esegui plot.py per generare i grafici.\n\n";
    return 0;
}
