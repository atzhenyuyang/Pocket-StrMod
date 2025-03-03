from pocket_strmod import Protein, Ligand, SplitPocket 
d = 10
a = Protein("./Pocket_StrMod/XXX.pdb")
l = Ligand("./Pocket_StrMod/XXX.sdf")
result_pdb,result_ligand = SplitPocket._split_pocket_with_surface_atoms(a,l,d)   # d in the residues which are within the distance with the ligand.
with open("./pockets/XXX10.pdb", "w") as f:
    f.write(result_pdb)

