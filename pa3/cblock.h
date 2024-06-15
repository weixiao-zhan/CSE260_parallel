typedef struct _control_block
{
    int m, n;
    int stats_freq;
    int plot_freq;
    int px, py;
    bool noComm;
    int niters;
    bool debug;
    bool wait;
} control_block;
// #define _MPI_ 1
// #define VERBOSE_PRINT 1