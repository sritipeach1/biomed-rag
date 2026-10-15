import os

# On Windows, the pip builds of PyTorch and FAISS each bundle their own OpenMP library, and loading
# both can stop the program with "OMP: Error #15". This setting is the standard workaround.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
