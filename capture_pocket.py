from pocket_strmod import Protein, Ligand, SplitPocket 
d = 10
a = Protein("/export/home/yangzhenyu/Pocket_StrMod/pockets/Mpro_8Y7T_new/8y7t_protein.pdb")
l = Ligand("/export/home/yangzhenyu/Pocket_StrMod/pockets/Mpro_8Y7T_new/8y7t_ligand.sdf")
result_pdb,result_ligand = SplitPocket._split_pocket_with_surface_atoms(a,l,d)   # d in the residues which are within the distance with the ligand.
with open("/export/home/yangzhenyu/Pocket_StrMod/pockets/Mpro_8Y7T_new/8Y7T_pocket_10.pdb", "w") as f:
    f.write(result_pdb)

