#x = '1AX1NAG' #+25
#x = '1A0JBEN' #same
#x = '1AIAPMP' #+30 (orlig and itreclig same)
#x = '1AKRFMN' #+35 (both recligs)
#x = '1AKUFMN' #+35 (both recligs)
#x = '1AL6OAA' #+110, itreclig+65
#x = '1AMQPMP' #same
#x = '1AX2NDG' #+80
#x = '1AX2XYP' #all same
#x = '1B4XMAE' #recligs -40
#x = '1B4XPLP' #+33, itreclig+15
#x = '1B4ZACT' #-35 ---
#x = '1B9VRA2' #+60, itreclig+10
#x = '1BDUFMT' #-13
#x = '1BG0ADP' #+25, iteclig-200
#x = '1BVI2GP' #+8
#x = '10GSVWW' #same, itreclig-320
#x = '17GSGTX' #+50
#x = '1B9TRAI' #+20
#x = '1C7EFMN' #both recligs +12
#x = '1C1XIPA' #-30
#x = '1AX1BGC' #same, itreclig+3
#x = '13GSSAS' #same
#x = '1AX2GAL' #all same
#x = '1AKTFMN' #+6, itreclig highest, + 8
#x = '1AXZBMA' #see if same ligand, similar score.
#x = '1BWKFMN' #both recligs nearly same score, higher than orlig. +23
#x = '1AKVFMN' #both recligs nearly same score, lot higher than orlig. +40
#x = '1BNUAL3' #intreclig lower than orlig. -14
x = '1B9SFDI' #+24
#x = '1BWCFAD' #high score, same. ~739
#x = '1B4BARG' #+20
#x = '1AX0A2G' #both recligs nearly same score, higher than orlig. +20
#x = '1AIQCXM' #+15
#x = ''

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

cmd.zoom(f'{x}_intact', buffer=3.5, state=0, complete=1)
cmd.show('wire', f'{x}rec')
cmd.color('gray70', f'{x}rec')
cmd.color('palecyan', f'{x}lig')
cmd.color('lightpink', f'{x}_intact')
util.cnc()