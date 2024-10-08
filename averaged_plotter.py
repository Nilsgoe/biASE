import matplotlib.pyplot as plt
import cmocean.cm as cmo
import numpy as np
import subprocess
import pandas as pd
import os 
import time

files = [file for file in os.listdir(os.getcwd()) if 'biase_c' in file and '.txt' in file and not '.dat' in file]
bin1 = 300
bin2=300
max1=1.2
max2=1.56
min1=0.3
min2=-0.35
scm = np.zeros(((bin1+1), (bin2+1)))
tcm = np.zeros(((bin1+1), (bin2+1)))
fes = np.zeros(((bin1+1), (bin2+1)))
c_4= np.zeros(((bin1+1), (bin2+1)))
c_5 = np.zeros(((bin1+1), (bin2+1)))
for file in files:
    o_file= f'fes_{file.split(".")[0]}.dat'
    # Reconstruct the free energy from HILLS using the plumed tool "sum_hills" :
    p = subprocess.Popen(f"plumed sum_hills --hills {file} --outfile {o_file}" 
                        f" --bin {bin1},{bin2} --min {min1},{min2} --max {max1},{max2}",
                        shell=True, stdout=subprocess.PIPE)
    p.wait()
    time.sleep(10)
    if fes.all()==0:
        with open(o_file, 'r') as f:
            header_lines = f.readlines()[:9]



    # Import free energy and reshape with the number of bins defined in the
    # reconstruction process.
    scm += np.loadtxt(o_file, usecols=0).reshape((bin1+1), (bin2+1))#was 201 everywhere
    tcm += np.loadtxt(o_file, usecols=1).reshape((bin1+1), (bin2+1))
    fes += np.loadtxt(o_file, usecols=2).reshape((bin1+1), (bin2+1))
    c_4 += np.loadtxt(o_file, usecols=3).reshape((bin1+1), (bin2+1))
    c_5 += np.loadtxt(o_file, usecols=4).reshape((bin1+1), (bin2+1))
    

scm /= len(files)
tcm /= len(files)
fes /= len(files)
c_4 /= len(files)
c_5 /= len(files)
# Write the arrays to a new file using the same format
output_file_path = "averaged_fes_biASE.dat"

# Create a DataFrame from the combinations
data = {
    'Column1': scm.flatten(),
    'Column2': tcm.flatten(),
    'Column3': fes.flatten(),
    'Column4': c_4.flatten(),
    'Column5': c_5.flatten()
}
#pd.options.display.float_format = '{:.8f}'.format
df = pd.DataFrame(data)
df = df.applymap(lambda x: f"{x:.{8}f}" if isinstance(x, float) else x)
# Write the DataFrame to the output file
data_content = df.to_csv(index=False, header=False,sep='\t')
header_content = ''.join(header_lines)

# Save the combined content to the output text file
with open(output_file_path, 'w') as f:
    f.write(header_content)
    f.write(data_content)
# Plot
fig, ax = plt.subplots(figsize=(10, 10))

# Plot free energy surface
im = ax.contourf(scm, tcm, fes, 10, cmap=cmo.tempo_r)
cp = ax.contour(scm, tcm, fes, 10,
                linestyles='-', colors='darkgray', linewidths=1.3)

# Plot parameters
#ax.set_xlabel('SCM', fontsize=40)
#ax.set_ylabel('TCM', fontsize=40)
ax.tick_params(axis='y', labelsize=32)
ax.tick_params(axis='x', labelsize=32)
cbar = fig.colorbar(im, ax=ax)
#cbar.set_label(label=r'FES[$\epsilon$]', fontsize=40)
cbar.ax.tick_params(labelsize=38)
#plt.scatter()
plt.tight_layout()
plt.savefig('averaged_fes_plot_biASE.png',dpi=300)
plt.show()
