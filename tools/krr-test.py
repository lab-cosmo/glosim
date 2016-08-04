#!/usr/bin/python
import sys
import numpy as np
import math
def main( kernel, trainvector, proplist,csi ):
       csi=float(csi)
      # kij=np.loadtxt(kernel)
       tc=np.loadtxt(trainvector)
       p=np.loadtxt(proplist)
      # kij = kij**csi
       #print lweight
       krp=[]
       with open(kernel) as f:
        last_pos = f.tell()
        header=f.readline().strip()
        if not header.startswith("#"): f.seek(last_pos)
        iframe=0
        for ln in f:
          ai=[float(x) for x in ln.split()]
          if (len(ai) != len(tc)):
            print "inconsistent kernel and train vector file"
            return
          t=0.0
          for j in range(len(tc)):
             t+=(ai[j]**csi)*tc[j]
          krp.append(t)
          print t,p[iframe],abs(t-p[iframe])
          iframe+=1

      # if (len(kij[0]) != len(tc)):
      #   print "inconsistent kernel and train vector file"
      #   return
      # if (len(kij) != len(p)):
      #   print "inconsistent kernel and property file"
      #   return
       #krp = np.dot(kij,tc)
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

