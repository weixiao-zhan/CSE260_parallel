[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/YNbQ13Wl)
Matrix Multiplication with CUDA

You will only need to edit files in src_todo folder.
Run all your commands from the root directory of the git repo.

sudo nvidia-smi -ac 5001,1590

To build the binary:
make -C build_T4 `cat src_todo_T4/OPTIONS.txt`

To build the binary with cublas:
make -C build_T4 cublastest=1 `cat src_todo_T4/OPTIONS.txt`

To clean the output files of the make command:
make -C build_T4 clean

To Run:
./mmpy `cat src_todo_T4/OPTIONS_RUNTIME.txt` -n 256

To run Script in tools folder:
./tools/run_ncu.sh

If you get Permission denied error when executing a file:
chmod +x name_of_file
eg: chmod +x tools/*

Find GPU chipset:
lspci | grep -i --color 'vga\|3d\|2d'
