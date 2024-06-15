import subprocess
import re

# Path to the header file
header_file_path = 'blislab/bl_config.h'

def modify_header_file(kc, mc, nc):
    # Read the original content
    with open(header_file_path, 'r') as file:
        lines = file.readlines()
    
    # Modify the lines with the new parameter values
    new_lines = []
    for line in lines:
        if line.startswith('#define DGEMM_KC'):
            new_lines.append(f'#define DGEMM_KC {kc}\n')
        elif line.startswith('#define DGEMM_MC'):
            new_lines.append(f'#define DGEMM_MC {mc}\n')
        elif line.startswith('#define DGEMM_NC'):
            new_lines.append(f'#define DGEMM_NC {nc}\n')
        else:
            new_lines.append(line)
    
    # Write the new content back to the file
    with open(header_file_path, 'w') as file:
        file.writelines(new_lines)

def compile_and_run():
    # Compile the program
    print("compiling...")
    subprocess.run(['make'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    # Run the benchmark and capture the output
    print("running...")
    result = subprocess.run(['./benchmark-blislab'], stdout=subprocess.PIPE, text=True)
    return result.stdout

def parse_output(output):
    # Extract the performance metric from the last line
    last_line = output.strip().split('\n')[-1]
    match = re.search(r'GeoMean\s+=\s+(\d+\.\d+)', last_line)
    if match:
        return float(match.group(1))
    else:
        return None

def gen_range(center, oneside_n, delta):
    # Calculate the start of the range
    start = center - oneside_n * delta
    # Calculate the end of the range (include one more step to ensure the last number is included)
    end = center + oneside_n * delta + 1
    # Generate and return the list using a list comprehension
    return list(range(start, end, delta))

# Main loop to iterate over parameters
if __name__ == "__main__":
    results = []
    for kc in gen_range(1536, 2, 64):
        for nc in gen_range(64, 2, 8):
            for mc in gen_range(1024, 2, 32):
                modify_header_file(kc, mc, nc)
                output = compile_and_run()
                geomean = parse_output(output)
                results.append((kc, mc, nc, geomean))
                print(f"KC: {kc}, MC: {mc}, NC: {nc}, GeoMean: {geomean}")

    # Optionally, save results to a file
    with open('results.csv', 'w') as file:
        file.write('DGEMM_KC,DGEMM_MC,DGEMM_NC,GeoMean\n')
        for kc, mc, nc, geomean in results:
            file.write(f"{kc},{mc},{nc},{geomean}\n")
