# Pocket_StrMod
A deep learning model for structure-based bioactivity optimization    
Bioactivity optimization is a crucial and highly skilled step in early drug discovery, traditionally accomplished through an iterative optimization of substituents, which is often time-consuming and costly. We herein propose a deep-learning model, Pocket-StrMod, for structure-based bioactivity optimization, which concurrently optimizes all the substituents by autoregressively generating atoms and covalent bonds at specified sites of a molecular scaffold located inside a protein pocket. The proposed model was applied to the bioactivity optimization of a SARS-CoV-2 main protease (Mpro) inhibitor, Hit1, which initially exhibited poor activity (IC50: 34.56 μM). Through two rounds of optimization, a total of six small molecules were selected to synthesize and test for bioactivity, ultimately yielding a potent compound, C5, with an IC50 value of 33.6 nM, a 1028-fold increase compared to Hit1. All the data obtained here highlights a great potential of deep learning in achieving fast and cheap bioactivity optimization in early drug discovery.

### Package Requirements:
Python 3.8.12 is used.
All the requirements are in requirements.txt, you can use 
~~~shell
pip install -r requirements.txt
~~~       
Or you can install one package by one package in the requirement.          
### Molecular Generating process
The molecular generating process mainly includes several steps:     
(1) Specify the protein pocket        
(2) Specify the molecular scaffold        
(3) Specify molecular growing sites       
(4) Set generation parameters     
(5) Run generate_molecule.py to generate molecules         

### Specify the protein Pocket
Run capture_pocket.py to get the protein pocket based on the ligand.       

### Specify the molecular number
To specify the growing sites on the scaffold, using molecule_number.py to show the number of atoms.
You should input a ".sdf" file of the molecule fragment.
After processing, you could get a png file.      
Based on this, you can designate which place to grow atoms and covalent bonds. Put the number as a list in --idx when generatiing molecules.

### Template match
Pocket_StrMod could include or exclude templates that users don't want to generate. We use SMARTS in mol_filter.py. For the users' convenience, users can get and test the SMARTS format in SMARTS_encoding_test.py to see if template matches.

### Command for molecular optimization
The molecule can be generated by running the following command, where the pocket pdb file, the molecular scaffold file, the generating position and the model parameter file are required, and the rest of the parameters are optional. You can use nohup command to generate molecules on the backend. An example of command is as follows"    
~~~shell
python generate_molecule.py -d cuda:0 -pkt ./pockets/Mpro_initial/Mpro_initial_pocket15.pdb -lig ./pockets/Mpro_initial/scaffold.sdf --idx "[12, 14, 15, 16]" --ckpt ./trained_model/parameter.pt -n 500 --name Mpro_optimization --path_name gen_results --lig_max 28 
~~~
The parameters of atoms and covalent bonds generation in ligand optimization:
~~~
usage: generate_molecule.py [-h] [-d DEVICE] [--ckpt CKPT] [-pkt POCKET] [-lig LIGAND] [--idx ATOM_Number] [-n generate_number] [--name NAME] [--lig_max lig_max][--print_SMILES print_SMILES] [--path_name path_name] [--readme README]

optional arguments:
  -h, --help            show this help message and exit
  -d DEVICE
                        specify the device used for molecular optimization, cuda:0/1/.../n or CPU.    
  --ckpt CKPT          
                        the model parameter file path
  -pkt POCKET
                        the pdb file of pocket in receptor
  -lig Ligand.sdf
                        the sdf file of the molecular scaffold
  -idx ATOM_Number      
                        the atom number on the scaffold, which are the specified sites for growing atoms and chemical bonds.                      
  -n generate_number
                        the number of molecules to be generated
  --file_name file_name           
                        the name of the file where the generated results are saved
  --path_name path_name
                        the file path for saving generation results
  --lig_max lig_max
                        the maximum atom number limitation of molecules which are generated
  --print_SMILES print_SMILES
                        whether print SMILES format in generative process
  --readme README, -rm README
                        description of the task

~~~



### Simpler instructions for ligand optimization    
You can run and change the hyper-parameters in generate_molecule.py to generate molecules, Please see generate.molecule.py for details.      

### Trained model
For users' convenience, we have put the trained model parameter in  ./trained_model/parameter.pt. There is no need for the users to train the model again.

### Model training
If you want to train the model, Model training needs a pretraining and finetuning process. You should prepare the dataset in .lmdb format.     

##### Pretraining Dataset
The pretraining dataset of Pocket_StrMod was selected from [ZINC 3D](https://zinc.docking.org/tranches/home/). You can use data_filter to filter out data. Use make_pretrain_data.py to obtain the selected dataset from ZINC 3D.

##### fine-tuning Dataset
The raw [CrossDocked2020]dataset could be downloaded here: (https://bits.csb.pitt.edu/files/crossdock2020/)       

##### Any Questions?       
If you have any questions, please raise a issue in the github.

