import os

def process_files(directory):
    # Get a list of all files in the directory
    files = [file for file in os.listdir(directory) if 'biase_c' in file and '.txt' in file]
    print(files)
    # Process each file
    for file in files:
        file_path = os.path.join(directory, file)
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Remove the first line from the file
        lines = lines[1:]

        # Insert the new three lines at the beginning of the file
        new_lines = ['#! FIELDS time c1.moment-2 c1.moment-3 sigma_c1.moment-2 sigma_c1.moment-3 height biasf\n',
         '#! SET multivariate false\n',
         '#! SET kerneltype stretched-gaussian\n']
        lines = new_lines + lines

        # Write the updated content back to the file
        with open(file_path, 'w') as f:
            f.writelines(lines)

cwd = os.getcwd()
process_files(cwd)

