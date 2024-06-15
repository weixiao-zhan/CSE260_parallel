#include <iostream>
#include "cblock.h"
#include "Plotting.h"
using namespace std;

extern control_block cb;

Plotter::Plotter()
{
    gnu_pipe = popen("gnuplot", "w");
}

void Plotter::updatePlot(double *U, int niter, int m, int n)
{
    double mx = -1.0e10, mn = 1.0e10;

    for (int i = 0; i < (m + 2) * (n + 2); i++)
    {
        if (U[i] > mx)
            mx = U[i];
        if (U[i] < mn)
            mn = U[i];
    }
    // fprintf(gnu_pipe, "\n\nunset key\n");
    fprintf(gnu_pipe, "set title \"niter = %d\n", niter);
    fprintf(gnu_pipe, "set xrange [0:%f]\n", m);
    fprintf(gnu_pipe, "set yrange [0:%f]\n", n);
    fprintf(gnu_pipe, "set size square\n");
    fprintf(gnu_pipe, "set key off\n");
    fprintf(gnu_pipe, "set pm3d map\n");
    fprintf(gnu_pipe, "set palette defined (-3 \"blue\", 0 \"white\", 1 \"red\")\n");

    // fprintf(gnu_pipe, "plot \"-\" with points lt 1 pt 10 ps 1\n");
    // Various color schemes
    // fprintf(gnu,"set palette rgbformulae 22, 13, 31\n");
    // fprintf(gnu,"set palette rgbformulae 30, 31, 32\n");

    // Write out the coordinates of the particles
    fprintf(gnu_pipe, "splot [0:%d] [0:%d][%f:%f] \"-\"\n", m - 2, n - 2, mn, mx);
    for (int i = 0; i < (m + 2) * (n + 2); i++)
    {
        int I = i / (n + 2);
        int J = i % (n + 2);
        fprintf(gnu_pipe, "%d %d %f\n", I, J, U[i]);
        if (J == n + 1)
            fprintf(gnu_pipe, "\n");
    }
    fprintf(gnu_pipe, "e\n");

    fflush(gnu_pipe);

    if (cb.wait)
    {
        cout << "Type any key to continue...\n";
        int dummy;
        cin >> dummy;
    }
}

Plotter::~Plotter()
{
    pclose(gnu_pipe);
}
