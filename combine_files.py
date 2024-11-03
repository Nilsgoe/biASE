import os
from ase.io import read, write

# Specify the output file name
output_file = "pathway.traj"

# Specify the pattern for your input files
input_pattern = "HCO_on_Rh_dis_biased_{}.traj"


# List to hold the first images from each file
images = []

# Loop over all the files
for i in range(-20,20):
    filename = input_pattern.format(i)
    
    if os.path.exists(filename):
        # Read the first image (index 0) from each .traj file
        image = read(filename, index=0)
        images.append(image)
    else:
        print(f"File {filename} not found!")

# Write all collected images to the output file
write(output_file, images)

print(f"First images from all traj files have been written to {output_file}.")

