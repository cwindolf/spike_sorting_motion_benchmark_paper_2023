from pathlib import Path
import getpass
import json
import os

import numpy as np
import matplotlib.pylab as plt



import MEArec as mr
import MEAutility as mu

import spikeinterface.full as si



# if you are not Sam nor Pierre you should add some lines here.
if getpass.getuser() == 'samuel.garcia':
    # sam working
    # base_folder = Path('/mnt/data/sam/DataSpikeSorting/mearec_drift_v6/')
    base_folder = Path('/mnt/data/sam/DataSpikeSorting/mearec_drift_v7/')

    si.Kilosort2_5Sorter.set_kilosort2_5_path('/home/samuel.garcia/Documents/SpikeInterface/code_sorters/Kilosort2.5/')
    # si.Kilosort3Sorter.set_kilosort3_path('/home/samuel.garcia/Documents/SpikeInterface/code_sorters/Kilosort3/')
    tmp_folder = base_folder / 'tmp'
    cell_folder = mr.get_default_cell_models_folder()
elif getpass.getuser() == 'cwindolf':
    base_folder = Path('/home/cwindolf/proj/measims')
    base_folder.mkdir(exist_ok=True)
    # cell_folder = Path('/home/cwindolf/proj/mearec_cells')
    # cell_folder.mkdir(exist_ok=True)
    cell_folder = mr.get_default_cell_models_folder()
    tmp_folder = Path(os.environ['TMPDIR'])
else:
    # pierre working
    base_folder = Path('/media/cure/Secondary/pierre/softwares/spikeinterface_drift_benchmarks/pierre')
    # TODO
    si.Kilosort2_5Sorter.set_kilosort2_5_path('/media/cure/Secondary/pierre/softwares/Kilosort-2.5/')
    # si.Kilosort3Sorter.set_kilosort3_path('/media/cure/Secondary/pierre/softwares/Kilosort-3.0/')
    tmp_folder = None
    cell_folder = mr.get_default_cell_models_folder()



## parameters for the recording generation

# this is for debugging for fast simulation
# probename = 'Neuronexus-32'
# this is for the final paper
probename = 'Neuropixels-128'

cells_positions = ['uniform', 'bimodal']
cells_rates = ['homogeneous', 'modulated']
drift_modes = ['rigid', 'non-rigid', 'bumps']
# cells_positions = ['uniform',]
# cells_rates = ['homogeneous', ]
# drift_modes = ['bumps',]



## parameters for estimation benchmark

localize_methods = ['center_of_mass', 'monopolar_triangulation', 'grid_convolution']
#localize_methods = ['monopolar_triangulation']
estimation_methods = ['decentralized', 'iterative_template']
#estimation_methods = ['decentralized', ]

interpolation_methods = ['kriging', 'idw', 'nearest', ]


