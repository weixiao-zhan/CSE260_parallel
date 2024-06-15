// Process command line arguments
// 
// Do not change the code in this file, as doing so
// could cause your submission to be graded incorrectly
//

#include <stdlib.h> // For: exit, drand48, malloc, free, NULL, EXIT_FAILURE
#include <stdio.h>  // For: perror
#include <string.h> // For: memset
#include <getopt.h>

void cmdLine(int argc, char *argv[], int& n, int& noCheck, int& identDebug, int& genDATA){
  // Command line arguments
  // Default value of the matrix size
  static struct option long_options[] = {
        {"no-check", no_argument, 0, 'c'},
        {"n", required_argument, 0, 'n'},
	{"i", no_argument, 0, 'i'},   // identiy matrix
      {"g", no_argument, 0, 'g'},   //genDATA
    };

 // Set default values
  n=0;
  noCheck = 0;
  identDebug = 0;
  genDATA = 0;
 // Process command line arguments
  int ac;
  for(ac=1; ac<argc; ac++) {
    int c;
    while ((c=getopt_long(argc,argv,"gcin:",long_options,NULL)) != -1){
      switch (c) {

      case 'c':
	noCheck = 1;
	break;
	  
      case 'i':
	// setup identity matrix x sequential matrix
	identDebug = 1;
	break;
	
      case 'n':
	// Size of the matrix
	n = atoi(optarg);
	break;

      case 'g':
      genDATA = 1;


      break;

	// Dont' check accuracy - used for tallying cache activity
	  // Error
      default:
	printf("Usage: mmpy [-n <matrix dim>]\n");
	exit(-1);
      }
    }
  }
}

