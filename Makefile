# Makefile — Integral Image C++17 + OpenMP
# ==========================================
# make           → build
# make run       → build + benchmark + plots
# make clean     → rimuovi binari e output

CXX      = g++
CXXFLAGS = -std=c++17 -O2 -march=native -fopenmp -Wall -Wextra
TARGET   = integral_image
SRC      = integral_image.cpp

WIDTH    = 4096
HEIGHT   = 4096
THREADS  = $(shell nproc)
RUNS     = 5

.PHONY: all run plots clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^
	@echo "  ✓  Compilato: $(TARGET)"

run: $(TARGET)
	./$(TARGET) $(WIDTH) $(HEIGHT) $(THREADS) $(RUNS)
	@$(MAKE) --no-print-directory plots

plots:
	python3 plot.py

clean:
	rm -f $(TARGET) results_*.csv plot_*.png
