import subprocess
import re

# Path to the header file
header_file_path = 'src_todo_T4/mytypes.h'

def modify_header_file(Bs, Ts):
    # Read the original content
    with open(header_file_path, 'r') as file:
        lines = file.readlines()
    
    # Modify the lines with the new parameter values
    new_lines = []
    for line in lines:
        if line.startswith('#define TILE_BLOCK_M'):
            new_lines.append(f'#define TILE_BLOCK_M {Bs}\n')
        elif line.startswith('#define TILE_BLOCK_K'):
            new_lines.append(f'#define TILE_BLOCK_K {Bs}\n')
        elif line.startswith('#define TILE_BLOCK_N'):
            new_lines.append(f'#define TILE_BLOCK_N {Bs}\n')
        elif line.startswith('#define TILE_THREAD_M'):
            new_lines.append(f'#define TILE_THREAD_M {Ts}\n')
        elif line.startswith('#define TILE_THREAD_N'):
            new_lines.append(f'#define TILE_THREAD_N {Ts}\n')
        else:
            new_lines.append(line)
    
    # Write the new content back to the file
    with open(header_file_path, 'w') as file:
        file.writelines(new_lines)

def compile():
    # Compile the program
    print("compiling...")
    subprocess.run(['make', '-C', 'build_T4'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

def run(n = 1024):
    # Run the benchmark and capture the output
    print("running...")
    result = subprocess.run(['./mmpy', '-n', f'{n}'], stdout=subprocess.PIPE, text=True)
    return result.stdout

def parse_output(output):
    # Regular expression to find the GFLOPS value
    match = re.search(r'\[(\d+\.\d+)\s+gflops\]', output)
    if match:
        return float(match.group(1))
    else:
        return None

# Main loop to iterate over parameters
if __name__ == "__main__":
    results = []
    for Bs in [16,32,64]:
        for Ts in [1,2,4,8]:
            modify_header_file(Bs,Ts)
            compile()
            for n in [127,128,129, 255,256,257, 511,512,513, 767,768,769, 
                      1023,1024,1025, 2047,2048,2049, 4095,4096,4097]:
                output = run(n)
                gflops = parse_output(output)
                results.append((Bs,Ts,n, gflops))
                print(f"Bs: {Bs}, Ts: {Ts}, N: {n}, Gflops: {gflops}")

    # Optionally, save results to a file
    with open('results.csv', 'w') as file:
        file.write('Bs,Ts,n,gflops\n')
        for Bs, Ts, n, gflops in results:
            file.write(f"{Bs},{Ts},{n},{gflops}\n")
