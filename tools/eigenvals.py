#!/usr/bin/python

import numpy as np
import sys

def main(fname):
   ffile=open(fname, "r")
   fline = ffile.readline()
   while fline[0]=='#': fline=ffile.readline()
   sline=map(float,fline.split())
   nel = len(sline)
   fmat = np.zeros((nel,nel), float)
   ik = 0
   while (len(sline)==nel):
      fmat[ik]=np.asarray(sline)
      fline = ffile.readline()
      sline=map(float,fline.split())
      ik+=1
   fmat = 0.5*(2-fmat*fmat)
   v = np.linalg.eigvalsh(fmat)
   print "finished reading"
   for i in range(len(v)):
        print i, v[i]

if __name__ == '__main__':
   main(*sys.argv[1:])
