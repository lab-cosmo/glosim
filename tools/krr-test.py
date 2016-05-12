#!/usr/bin/python
import sys
import numpy as np
import math
def main( kernel, trainvector, proplist,csi ):
       csi=float(csi)
       kij=np.loadtxt(kernel)
       tc=np.loadtxt(trainvector)
       p=np.loadtxt(proplist)
       kij = kij**csi
       #print lweight
       if (len(kij[0]) != len(tc)):
         print "inconsistent kernel and train vector file"
         return
       if (len(kij) != len(p)):
         print "inconsistent kernel and property file"
         return
       krp = np.dot(kij,tc)
       nconf=len(p)
       mae=abs(krp[:]-p[:]).sum()/nconf
       rms=np.sqrt(((krp[:]-p[:])**2).sum()/nconf)
       diff=abs(krp[:]-p[:])
       sup=max(diff)
       print "# test-set MAE: %f RMS: %f SUP %f" % (mae, rms, sup)
       comment= "# test-set MAE: "+str(mae)+ " RMSE: "+ str(rms)+" SUP: " +str(sup)
       fname=proplist+".predict"
       f=open(fname,'w')
       np.savetxt(f,krp,header=comment)
       f.close()


if __name__ == '__main__':
   main(*sys.argv[1:])

