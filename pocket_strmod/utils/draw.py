import os
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import  Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
from .model_training import verify_dir_exists
from PIL import Image
import matplotlib.pyplot as plt

def draw_docked_mol_list(mol_list, molsPerRow=5, subImgSize=(300,300)):
    opts = DrawingOptions()
    opts.atomLabelFontSize = 30
    opts.bondLineWidth = 1.5
    opts.colorBonds = False
    
    legends = []
    for m in mol_list:
        AllChem.Compute2DCoords(m)
        docking_score = 'Docking Score: {:.3f}'.format(float(m.GetProp('r_i_docking_score')))
        mw = 'MolWt: {:.3f}'.format(Descriptors.MolWt(m))
        name = m.GetProp('_Name')
        legend = name + '\n' + mw + '\n' + docking_score
        legends.append(legend)

    img = Draw.MolsToGridImage(
        mol_list,
        molsPerRow=molsPerRow,
        subImgSize=subImgSize,
        legends=legends,
        returnPNG=False
    )
    return img


def HighlightAtomByWeights(mol, save=None, size=(300,300), colors=['#FFFFFF','#FF0000'], bondLineWidth=5, FontSize=3,fixedBondLength=50,    
                           legend=None, legendFontSize=5, elemColor=False, withIsomeric=True): 
    draw = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    option = rdMolDraw2D.MolDrawOptions()
    option.bondLineWidth = bondLineWidth
    option.fixedBondLength = fixedBondLength
    option.setHighlightColour((0.95,0.7,0.95))
    option.baseFontSize = FontSize
    option.legendFontSize = legendFontSize
    if elemColor == False:
        option.updateAtomPalette({k: (0, 0, 0) for k in DrawingOptions.elemDict.keys()})
    draw.SetDrawOptions(option)
    if withIsomeric:
        AllChem.Compute2DCoords(mol)
    else:
        smi = Chem.MolToSmiles(mol, isomericSmiles=False)
        mol = Chem.MolFromSmiles(smi)
    rdMolDraw2D.PrepareAndDrawMolecule(draw, mol, legend=legend)
    draw.FinishDrawing()
    if save:
        if '.png' in save:
            n = 0
            path = '/'.join(save.split('/')[0:-1])
            while os.path.exists(save):
                n += 1
                name = save.split('/')[-1].split('.')[0]
                name = name + '_' + str(n) + '.png'
                save = path + '/' + name
            else:
                draw.WriteDrawingText(save)
    else:
        return draw

def CombineImages(img_file_list, col_num=7, save_dir='./', title='image', img_size=None):
    
    num_img = len(img_file_list)
    num_row = num_img//col_num+1 if num_img%col_num != 0 else num_img//col_num
    
    if img_size is None:
        img_size = Image.open(img_file_list[0]).size
    toImage = Image.new('RGB', (img_size[1]*col_num, img_size[0]*num_row), color=(255,255,255))
    x_cusum = 0
    y_cusum = 0
    num_has_paste = 0
    for img_file in img_file_list:
        img = Image.open(img_file)
        #print((x_cusum, y_cusum))
        toImage.paste(img, (x_cusum, y_cusum))
        num_has_paste += 1
        #print(num_has_paste)
        if num_has_paste%col_num==0:
            x_cusum = 0
            y_cusum += img_size[1]
        else:
            x_cusum += img_size[0]
            
    plt.xticks([])
    plt.yticks([])
    plt.axis('off') 
    plt.imshow(toImage)
    
    verify_dir_exists(save_dir)
    toImage.save(save_dir+'/'+title+'.png')
    plt.clf()    