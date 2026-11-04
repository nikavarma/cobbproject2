import numpy as np
import pandas as pd
import scipy
import matplotlib
import sklearn
import prody
import MDAnalysis as mda

from prody import parsePDB
from pathlib import Path

#everything below this involves setting the trajectories:


cwd = Path.cwd()
print(cwd)

import numpy as np
import MDAnalysis as mda


ref_path = cwd / "4AKE_Monomer"/ "4AKE_Monomer_nowat.pdb"
mobile_path = cwd/ "1AKE_AP5" / "1AKE_AP5_nowat.pdb"
ref = mda.Universe(ref_path)
mobile = mda.Universe(mobile_path)
CA = "protein and name CA"
mobile_ca, ref_ca = mobile.select_atoms(CA), ref.select_atoms(CA)
assert np.array_equal(mobile_ca.resids, ref_ca.resids)
assert np.array_equal(mobile_ca.icodes, ref_ca.icodes)
assert np.array_equal(mobile_ca.resnames, ref_ca.resnames)
assert np.array_equal(mobile_ca.names, ref_ca.names)


print("\nOutput1:")
print("Matched C-alpha atoms:", len(mobile_ca))
print("Coordinate shape:", mobile_ca.positions.shape)
print("\n")


from MDAnalysis.analysis import align, rms

CORE = CA + " and (resid 1:29 or resid 60:121 or resid 160:214)"
align.alignto(mobile.select_atoms(CORE), ref.select_atoms(CORE),
select=None, weights=None, match_atoms=False)
value = rms.rmsd(mobile_ca.positions, ref_ca.positions,
center=False, superposition=False)
print("\nOutput2:")
print(f"CORE-fit / all-C-alpha RMSD: {value:.3f} angstrom") #corefit / all-c-alpha RMSD?? tf does this mean


topology_path = cwd / "1AKE_AP5" / "1AKE_AP5_nowat.psf"
trajectory_path = cwd / "1AKE_AP5" / "1AKE_AP5_nowat.dcd"
print(topology_path)

u = mda.Universe(topology_path, trajectory_path, dt=2400.0)

print("\nOutput3: ")
print("Atoms:", u.atoms.n_atoms)
print("Saved frames:", len(u.trajectory))
print("Known saved spacing (ps):", u.trajectory.dt)
print("One frame's coordinates:", u.atoms.positions.shape)



import pandas as pd
import matplotlib.pyplot as plt



# PDB masses are guessed; copy the trajectory's CA masses so the mass check passes
ref.select_atoms(CA).masses = u.select_atoms(CA).masses

# u: matched trajectory; ref, CA, CORE and rms: as above
result = rms.RMSD(u, ref, select=CORE, groupselections=[CA], ref_frame=0, weights=None).run()
table = pd.DataFrame(result.results.rmsd,
columns=["frame", "time_ps", "CORE_RMSD_A", "all_CA_RMSD_A"])
table["time_ns"] = (table["time_ps"] - table["time_ps"].iloc[0]) / 1000
fig, ax = plt.subplots()
ax.plot(table["time_ns"], table["all_CA_RMSD_A"])
ax.set(title="All Cα RMSD to 4AKE after CORE fitting",
xlabel="Relative time (ns)", ylabel="RMSD (Å)")

table_path = cwd / "tables" / "table1.csv"
figure_path = cwd / "figures" / "figure1.png"



table.to_csv(table_path, index=False)
fig.savefig(figure_path, dpi=300, bbox_inches="tight")


#after setting the trajectories we can do some analyses to understand more details about how e coli's adenylate kinase moves. 
