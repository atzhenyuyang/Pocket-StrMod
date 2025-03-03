from rdkit import Chem


def show_atom_number(mol, label):
    for atom in mol.GetAtoms():
        atom.SetProp(label, str(atom.GetIdx()))
    return mol

def split_bond(mol, bond_atom_idx):      
    rw_mol = Chem.RWMol()
    for a in mol.GetAtoms():
        rw_mol.AddAtom(Chem.Atom(a.GetAtomicNum()))

    for b in mol.GetBonds():
        start = b.GetBeginAtom()
        start_idx = start.GetIdx()
        end = b.GetEndAtom()
        end_idx = end.GetIdx()
        if start_idx in bond_atom_idx and end_idx in bond_atom_idx:  
            continue
        else:
            rw_mol.AddBond(start_idx, end_idx, b.GetBondType())

    rd_mol = rw_mol.GetMol()
    smi = Chem.MolToSmiles(rd_mol)
    return smi


def substituent_diversity(sdf_file, bond_atom_idx):
    supplier = Chem.SupplierFromFilename(sdf_file)
    substituent_list_1 = []
    for m in supplier:
        smi = split_bond(m, bond_atom_idx)   
        if '.' not in smi: continue
        substituent = sorted(
            [(len(i), i) for i in smi.split('.')], 
            key=lambda x:x[0]
            )[0]                             
        substituent_list_1.append(substituent[1])   
    return set(substituent_list_1)


def split_bond_1(mol, atom_idx, skeleton_idx=set(range(18))):
    rw_mol = Chem.RWMol()
    for a in mol.GetAtoms():
        rw_mol.AddAtom(Chem.Atom(a.GetAtomicNum()))
    for b in mol.GetBonds():
        start = b.GetBeginAtom()
        start_idx = start.GetIdx()
        end = b.GetEndAtom()
        end_idx = end.GetIdx()
        if atom_idx in [start_idx, end_idx] and not {start_idx, end_idx}.issubset(skeleton_idx):
            continue
        else:
            rw_mol.AddBond(start_idx, end_idx, b.GetBondType())

    rd_mol = rw_mol.GetMol()
    smi = Chem.MolToSmiles(rd_mol)
    return smi



def substituent_diversity_1(sdf_file, atom_idx, skeleton_idx=set(range(18))):
    supplier = Chem.SupplierFromFilename(sdf_file)
    substituent_list_1 = []
    for m in supplier:
        smi = split_bond_1(m, atom_idx, skeleton_idx=skeleton_idx)
        if '.' not in smi: continue
        substituent = sorted(
            [(len(i), i) for i in smi.split('.')], 
            key=lambda x:x[0]
            )[0]
        substituent_list_1.append(substituent[1])
    return set(substituent_list_1)


num_substituent_site_1 = substituent_diversity('./molecules_generated/generated_round_1.sdf', {4,7})
num_substituent_site_2 = substituent_diversity_1('./molecules_generated/generated_round_1.sdf', 15, skeleton_idx=set(range(18)))
num_substituent_site_3 = substituent_diversity_1('./molecules_generated/generated_round_1.sdf', 16, skeleton_idx=set(range(18)))
num_substituent_site_4 = substituent_diversity_1('/./molecules_generated/generated_round_1.sdf', 14, skeleton_idx=set(range(18)))
combination_1 =  num_substituent_site_1 | num_substituent_site_2
combination_2 =  combination_1 | num_substituent_site_3
combination_3 =  combination_2 | num_substituent_site_4
print(len(combination_3))