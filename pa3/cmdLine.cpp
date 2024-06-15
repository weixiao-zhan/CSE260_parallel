// Process command line arguments
//
//
// Do not change the code in this file, as doing so
// could cause your submission to be graded incorrectly
//
#include <assert.h>
#include <getopt.h>
#include <stdlib.h>
#include <iostream>
#include "cblock.h"
#ifdef _MPI_
#include <mpi.h>
#endif
#include "omp.h"

using namespace std;

void Stop();
extern control_block cb;

void cmdLine(int argc, char *argv[])
{
    /// Command line arguments
    // Default value of the domain sizes
    static struct option long_options[] = {
        {"n", required_argument, 0, 'n'},
        {"niters", required_argument, 0, 'i'},
        {"stats-freq", required_argument, 0, 's'},
        {"plot", required_argument, 0, 'p'},
        {"px", required_argument, 0, 'x'},
        {"py", required_argument, 0, 'y'},
        {"nocomm", no_argument, 0, 'k'},
        {"debug", no_argument, 0, 'd'},

    };
    int nprocs = 1, myrank = 0;
#ifdef _MPI_
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
#endif

    // Process command line arguments
    for (int ac = 1; ac < argc; ac++)
    {
        int c;
        while ((c = getopt_long(argc, argv, "n:i:s:x:y:p:kd", long_options, NULL)) != -1)
        {
            switch (c)
            {

                // Size of the computational box
            case 'n':
                cb.n = atoi(optarg);
                break;
                //
                // X processor geometry
            case 'x':
                cb.px = atoi(optarg);
                break;

            // X processor geometry
            case 'y':
                cb.py = atoi(optarg);
                break;

                // # of iterations
                // Use this option control the number of mesh sweeps
            case 'i':
                cb.niters = atoi(optarg);
                break;

                // Print statistics for assessing correctness
            case 's':
                cb.stats_freq = atoi(optarg);
                break;

                // Plot the excitation variable
            case 'p':
                cb.plot_freq = atoi(optarg);
                break;

            // Debug ouput
            case 'd':
                cb.debug = true;
                break;

            // Shut off communication
            case 'k':
                cb.noComm = true;
                break;

                // Error
            default:
                cout << "Usage: apf [-n <domain size>] [-i <# iterations>]";
                cout << "\n\t    ";
                cout << " [-s <stats frequency>[-p <plot frequency>]\n\t";
                cout << "     [-x <x processor geometry>]";
                cout << "     [-y <x processor geometry>]";
                cout << "     [-d <debug>]";
                cout << "     [-k <no communication>]" << endl;
                cout << endl;
                exit(-1);
            }
        }
    }
    if ((cb.plot_freq > 0) && cb.debug)
    {
        cb.wait = true;
    }
#ifdef _MPI_
    if ((cb.px * cb.py) != nprocs)
    {
        if (!myrank)
        {
            if ((cb.px * cb.py) > nprocs)
                cout << "\n *** The number of processes in the geometry (" << cb.px * cb.py << ")\n     is larger than the number of available cores (" << nprocs << ")" << endl
                     << endl;
            else
                cout << "\n *** The number of processes in the geometry (" << cb.px * cb.py << ")\n     is smaller than the number of avilable cores (" << nprocs << ")" << endl
                     << endl;
        }
        Stop();
    }
#else
    if ((cb.px * cb.py) > 1)
    {
        if (!myrank)
            cout << "\n *** The number of processes in the geometry > 1, but you have not enabled MPI\n";
        Stop();
    }
#endif
}
