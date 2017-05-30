#!/usr/bin/env python
"""
Just a minimal stub to compute SOAP vectors from a list of structures and print them out on a file.
"""

import quippy
import sys, time, ast
from multiprocessing import Process, Value, Array
import argparse
from random import randint
from libmatch.environments import alchemy, environ
from libmatch.structures import structk, structure, structurelist
import os
import numpy as np
from copy import copy 
from time import ctime
from datetime import datetime
import gc
import cPickle as pickle
import code

# tries really hard to flush any buffer to disk!
def flush(stream):
    stream.flush()
    os.fsync(stream)
def atomicno_to_sym(atno):
  pdict={1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm', 101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Ha', 106: 'Sg', 107: 'Ns', 108: 'Hs', 109: 'Mt', 110: 'Unn', 111: 'Unu'}   
  return pdict[atno]


def main(filename, nmax, lmax, coff, cotw, gs, centerweight, prefix=""):
    start_time = datetime.now()
    
    filename = filename[0]
    # sets a few defaults
    if prefix=="": prefix=filename
    if prefix.endswith('.xyz'): prefix=prefix[:-4]
    prefix=prefix+"-n"+str(nmax)+"-l"+str(lmax)+"-c"+str(coff)+"-g"+str(gs)

    print >> sys.stderr, "using output prefix =", prefix
    # Reads input file using quippy
    print >> sys.stderr, "Reading input file", filename
    (first,last)=(0,0); # reads the whole thing 
    if first==0: first=None; 
    if last==0: last=None
    al = quippy.AtomsList(filename, start=first, stop=last);
    print >> sys.stderr, len(al.n) , " Configurations Read"
    
    # determines "kit" (i.e. max n and kind of atoms present)
    spkit = {}
    for at in al:
        atspecies = {}
        for z in at.z:      
            if z in atspecies: atspecies[z]+=1
            else: atspecies[z] = 1
            
        for (z, nz) in atspecies.iteritems():
            if z in spkit:
                if nz>spkit[z]: spkit[z] = nz
            else:
                spkit[z] = nz
    
    # species string 
    zsp=spkit.keys();
    zsp.sort(); 
    lspecies = 'n_species='+str(len(zsp))+' species_Z={ '
    for z in zsp: lspecies = lspecies + str(z) + ' '
    lspecies = lspecies + '}'   
    lspecies='n_species=3 species_Z={1 6 7} ' 
    print lspecies
    fout=open(prefix+".soap","w")
    for at in al:
        fout.write("%d\n" % (len(at.z)))
        at.set_cutoff(coff);
        at.calc_connect();
        nel = np.bincount(at.z)
        
        # ok, it appears you *have* to do this atom by atom. let's do that, but then re-sort in the same way as in the input
        soaps = {}
        sz = {}
        for z in at.z: 
            if z in sz: sz[z]+=1
            else: sz[z]=1
        for (z, nz) in sz.iteritems():
            soapstr=("soap central_reference_all_species=F central_weight="+str(centerweight)+
           "  covariance_sigma0=0.0 atom_sigma="+str(gs)+" cutoff="+str(coff)+" cutoff_transition_width="+str(cotw)+
           " n_max="+str(nmax)+" l_max="+str(lmax)+' '+lspecies+' Z='+str(z))
            desc = quippy.descriptors.Descriptor(soapstr )
            print soapstr
            soaps[z] = desc.calc(at)["descriptor"]
        for z in at.z:
            fout.write("%3s  " % (atomicno_to_sym(z)))
            np.savetxt(fout, [ soaps[z][len(soaps[z])-sz[z]] ])
            sz[z] -=1
        
        print sz

    fout.close() 

if __name__ == '__main__':
      parser = argparse.ArgumentParser(description="""Computes the similarity matrix between a set of atomic structures 
                           based on SOAP descriptors and an optimal assignment of local environments.""")
                           
      parser.add_argument("filename", nargs=1, help="Name of the LibAtom formatted xyz input file")
      parser.add_argument("-n", type=int, default='8', help="Number of radial functions for the descriptor")
      parser.add_argument("-l", type=int, default='6', help="Maximum number of angular functions for the descriptor")
      parser.add_argument("-c", type=float, default='5.0', help="Radial cutoff")
      parser.add_argument("--cotw", type=float, default='0.5', help="Cutoff transition width")
      parser.add_argument("-g", type=float, default='0.5', help="Atom Gaussian sigma")
      parser.add_argument("-cw", type=float, default='1.0', help="Center atom weight")
      parser.add_argument("--prefix", type=str, default='', help="Prefix for output files (defaults to input file name)")
      
           
      args = parser.parse_args()

      main(args.filename, nmax=args.n, lmax=args.l, coff=args.c, cotw=args.cotw, gs=args.g, centerweight=args.cw, prefix=args.prefix)
