'''
Script to open pymol with original and reconstructed ligands imported and custom colored.
'''


import __main__
__main__.pymol_argv = [ 'pymol' ]

import pymol

pymol.finish_launching()

from pymol import cmd, util

cmd.window()
cmd.bg_color('white')
prt = cmd.load(f'/projects/mai/users/kkxw544_magdalena/deepfrag_data/{x}rec.pdb')
lgd = cmd.load(f'/projects/mai/users/kkxw544_magdalena/deepfrag_data/{x}lig.sdf')
mlc = cmd.load(f'/projects/mai/users/kkxw544_magdalena/deepfrag_data/{x}_intact.sdf')
mlc = cmd.load(f'/projects/mai/users/kkxw544_magdalena/deepfrag_data/{x}_mult.sdf')

cmd.zoom(f'{x}_intact', buffer=3.5, state=0, complete=1)
cmd.show('wire', f'{x}rec')
cmd.color('gray70', f'{x}rec')
cmd.color('palecyan', f'{x}lig')
cmd.color('lightpink', f'{x}_intact')
util.cnc()