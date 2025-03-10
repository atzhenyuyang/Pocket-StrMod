from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import  Draw
from rdkit.Chem.Draw.MolDrawing import  DrawingOptions 

def show_atom_number(mol, label):
    for atom in mol.GetAtoms():
        atom.SetProp(label, str(atom.GetIdx()))
    return mol

opts = DrawingOptions()
opts.includeAtomNumbers=True
m = Chem.MolFromMolFile("/export/home/yangzhenyu/Pocket_StrMod/pockets/Mpro_8Y7T_new/8y7t_ligand.sdf", sanitize=1) 
AllChem.Compute2DCoords(m)
show_atom_number(m, 'atomLabel')
opts.includeAtomNumbers=True
opts.bondLineWidth=2.8
image = Draw.MolToImage(m, options = opts, size=(500,500))
image.save("/export/home/yangzhenyu/Pocket_StrMod/pockets/Mpro_8Y7T_new/8y7t.png")