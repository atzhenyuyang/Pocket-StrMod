from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import  Draw
from rdkit.Chem.Draw.MolDrawing import  DrawingOptions 

m = Chem.MolFromSmiles("O=C(N1C(C)C)N(C2=CC=CC=C2)N=C(C#N)C1=O") 
m_1 = Chem.MolFromSmiles("O=C(N1C2=CC=CC=C2)N(C3=CC=CC=C3)N=C(C#N)C1=O")
a = Chem.MolToSmarts(m)
print(a)
pattern = Chem.MolFromSmarts('[#8]=[#6]1~&@[#7](-[#6](~[#6,#7,#8,#9,#15,#16,#17,#35,#53])~[#6])~&@[#6](~&@[#6](~&@[#7]~&@[#7]~&@1-[#6]1~&@[#6]~&@[#6]~&@[#6]~&@[#6]~&@[#6]~&@1)-[#6]#[#7])=[#8]')
print(m_1.GetSubstructMatch(pattern))
print(m_1.HasSubstructMatch(pattern))


def show_atom_number(mol, label):
    for atom in mol.GetAtoms():
        atom.SetProp(label, str(atom.GetIdx()))
    return mol

opts = DrawingOptions()
opts.includeAtomNumbers=True
AllChem.Compute2DCoords(pattern)
show_atom_number(pattern, 'atomLabel')
opts.includeAtomNumbers=True
opts.bondLineWidth=2.8
image = Draw.MolToImage(pattern, options = opts, size=(500,500))
image.save("1.png")


