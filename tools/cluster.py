#!/usr/bin/python
import argparse
import numpy as np
import sys
import numpy as np
import scipy.cluster.hierarchy as sc
import itertools
from scipy.stats import kurtosis,skew
from scipy.stats.mstats import kurtosistest
from os.path import basename
try:
  from matplotlib import pyplot as plt
except:
  print "matplotlib is not available. You will not be able to plot"

from collections import Counter

def main(distmatrixfile,dcut,mode='average',proplist='',plot=False,calc_sd=False,rect_matrixfile='',mtcia=False,verbose=False):
   project=False
   sim=np.loadtxt(distmatrixfile)
   if rect_matrixfile != '' :
      rect_matrix=np.loadtxt(rect_matrixfile)
      if (len(sim) != len(rect_matrix[0])):
         print "Inconsistent dimesion of rect matrix file"
         return
      project=True
   if proplist!='': prop=np.loadtxt(proplist)
   sim2=sim*sim
   Z=sc.linkage(sim2,mode)
   if mtcia : mathematica_cluster(Z,sim)
   n=len(sim)
   cdist=Z[:,2]
   if verbose : 
     np.savetxt('linkage.dat',Z)
     np.savetxt('dist.dat',cdist)
   nclust=estimate_ncluster(cdist,dcut)
   print "Estimated ncluster:",nclust
   print "mean+std, cutoffdist", np.sqrt(np.var(Z[:,2]))+np.mean(Z[:,2]),(Z[n-nclust,2])
   clist=sc.fcluster(Z,nclust,criterion='maxclust')
   c_count=Counter(clist)
   print "Number of clusters", len(c_count)
   print "nconfig     meand    variance   rep_config  Kurtosis skewness  Multimodal "
   rep_ind=[]
   structurelist=[]
   for iclust in range(1,len(c_count)+1):  #calculate mean dissimilary and pick representative structure for each cluster
      indices = [i for i, x in enumerate(clist) if x == iclust] #indices for cluster i
      nconf=len(indices)
      structurelist.append(indices)
      sumd=0.0
      kurt=0.0
      skewness=0.0
      multimodal=False
      if proplist!='' and len(indices)>1 :
            kurt= kurtosis(prop[indices],fisher=False)
            skewness=skew(prop[indices])
            if kurt !=0 : modc=(skewness*skewness+1)/kurt
            if modc >(5.0/9.0) :
               multimodal=True
      #calculate mean dissimilarity in each group
      for iconf in range(len(indices)):
         ind1=indices[iconf]
         for jconf in range(len(indices)):
           ind2=indices[jconf]
           sumd+=sim2[ind1][ind2]
      meand=np.sqrt(sumd/(nconf*nconf))
      
      # pick the configuration with min variance in the group
      minvar=9999
      var=0.0
      for iconf in range(len(indices)):
        ivar=0.0
        ind1=indices[iconf]
        for jconf in range(len(indices)):
          ind2=indices[jconf]
          ivar+=(sim2[ind1][ind2]-meand**2)**2
        ivar=ivar/nconf
        var+=ivar  
        if(ivar<minvar):  
          minvar=ivar
          iselect=ind1
      var=var/nconf  
      rep_ind.append(iselect)
      print nconf, meand , np.sqrt(var) , iselect ,kurt, skewness, multimodal
#   print rep_ind
   print "index of clusters"
   for i in structurelist:
      print "index=",i
   filename=basename(distmatrixfile)+'-cluster.index'
   f=open(filename,"w")
   f.write("groupid representative \n ")
   for i in range(len(sim)):
      iselect=0
      if i in rep_ind: iselect=2
      f.write("%d   %d \n " %(clist[i],  iselect)) 
   f.close()
   if(project):
     project_groupid,project_rep=project_config(clist,rect_matrix,rep_ind)
     filename=basename(rect_matrixfile)+'-cluster.index'
     f=open(filename,"w")
     f.write("groupid representative \n ")
     for i in range(len(project_groupid)):
        iselect=0
        if i in project_rep: iselect=2
        f.write("%d   %d \n " %(project_groupid[i],  iselect)) 
     f.close()
   if (calc_sd):
     filename= filename=basename(distmatrixfile)+'-cluster-sd.dat'
     f=open(filename,"w")
     f.write("dist_sd ")
     if proplist!='':f.write("prop_sd ")
     f.write("representative config")
     f.write("\n")
     sim_sd,rep_index=dissimilarity_sd(Z,sim2) 
     if proplist!='': psd=prop_sd(Z,prop,verbose)
     for i in range(len(Z)):
         f.write("%f" %(sim_sd[i]))
         if proplist!='':f.write("   %f" %(psd[i]))
         f.write("  %d" %(rep_index[i]))
         f.write("\n")
   if plot: 
        filename=basename(distmatrixfile)+'-dendogram.eps'
        plotdendro(Z,nclust,filename,rep_ind)

def plotdendro(Z,ncluster,filename,rep_ind):
  plt.figure(figsize=(10, 15))
  plt.title('Hierarchical Clustering Dendrogram')
  plt.xlabel('sample index')
  plt.ylabel('distance')
  d=sc.dendrogram(Z,truncate_mode='lastp', p=ncluster,orientation='right',leaf_rotation=90.,leaf_font_size=20.,show_contracted=False)
#  coord = np.c_[np.array(d['icoord'])[:,1:3],np.array(d['dcoord'])[:,1]]
#  coord = coord[np.argsort(coord[:,2])]
  num=ncluster-1
  coord=[]
  for i in range(len(d['icoord'])):
    if d['dcoord'][i][0]==0.0 :
     coord.append(d['icoord'][i][0])
  for i in range(len(d['icoord'])):
    if d['dcoord'][i][3]==0.0 :
     coord.append(d['icoord'][i][3])
  #print d['leaves']
  #return
  #for posi in coord:
  # x = posi
  #  y = 0.05
  #  plt.plot(x, y, 'ro')
  #  plt.annotate("%2i" % rep_ind[num], (x, y), xytext=(0, -8),
  #               textcoords='offset points',
  #               va='top', ha='center')
  #  num = num-1
  #plt.show()
  
  plt.savefig(filename, dpi=100, facecolor='w', edgecolor='w',
        orientation='portrait', papertype='letter', format=None,
        transparent=True, bbox_inches=None, pad_inches=0.1,
        frameon=None)
      
def project_config(clusterlist,rect_matrix,rep_ind):
  nland=len(rect_matrix[0])
  if nland != len(clusterlist) : 
     print "Dimension Mismatch for rect matrix" 
     stop 
  n=len(rect_matrix)
  groupid=[]
  for i in range(n):
    mind=10
    for j  in range(nland): # find which cluster it belongs 
        d=rect_matrix[i][j]
        if d <mind : 
            mind=d #find min distance config from config from  all clusters 
            icluster_select=clusterlist[j]
    groupid.append(icluster_select)
  project_rep=[]
  for iconfig in rep_ind: 
    mind=np.min(rect_matrix[:,iconfig])
    if (mind <1E-9):
      iselect=np.argmin(rect_matrix[:,iconfig])               
      project_rep.append(iselect)
  return(groupid,project_rep)
def mathematica_cluster(Z,sim):
   n=len(sim)
   clusterlist=[]
   nlist=[]
   for i in range(len(Z)):
    id1=int(Z[i,0])
    id2=int(Z[i,1])
    if((id1 < n) and (id2<n)):  # when two configurations are merged note their index
       #clusterlist.append('Cluster'+str([id1,id2,Z[i,2],1,1]))
       clusterlist.append([id1,id2,Z[i,2],1,1])
       nlist.append(2)
       #ncluster+=1
    else:
      cl=[]
      icount=0
      if id1>=n:  # this means merging is happening with previously formed cluster
        icluster=int(id1)-n
   #     for x in clusterlist[icluster]: #we already have the list for the old cluster
        cl.append(clusterlist[icluster])
        n1=nlist[icluster]
      else:
        cl.append(id1)
        n1=1
      if id2>=n: # same logic as before
        icluster=int(id2)-n
       # for x in clusterlist[icluster]:
        cl.append(clusterlist[icluster])
        n2=nlist[icluster]
      else:
        cl.append(id2)
        n2=1
      cl.append(Z[i,2])
      cl.append(n1)
      cl.append(n2)
   #   tmp='Cluster'+str(cl)
   #   tmp.replace("'","")
   #   clusterlist.append(tmp)
      clusterlist.append(cl)
      nlist.append(n1+n2)
   f=open('cluster-mathematica.dat','w')
#   clusterlist[n-2].replace("'","")
#   clusterlist[n-2].replace("\\","")
   f.write("%s" %str(clusterlist[n-2]))
   f.close()
   return  
def dissimilarity_sd(Z,sim):
  n=len(sim)
  clusterlist=[]
  ncluster=0
  sdlist=[]
  rep_index=[]
  for i in range(len(Z)):
    id1=int(Z[i,0])
    id2=int(Z[i,1])
    if((id1 < n) and (id2<n)):  # when two configurations are merged note their index
       clusterlist.append([id1,id2])
       ncluster+=1
    else:
      cl=[]
      icount=0
      if id1>=n:  # this means merging is happening with previously formed cluster
        icluster=int(id1)-n
        for x in clusterlist[icluster]: #we already have the list for the old cluster
          cl.append(x)
      else:cl.append(id1)
      if id2>=n: # same logic as before
        icluster=int(id2)-n
        for x in clusterlist[icluster]:
          cl.append(x)
      else:cl.append(id2)
      clusterlist.append(cl) # append the index of the members at this stage of clustering 
#   calculate mean dissimilarity of the cluster
    sumd=0.0
    icount=0
    for iconf in range(len(clusterlist[i])):
        ind1=clusterlist[i][iconf]
        for jconf in range(iconf):
          ind2=clusterlist[i][jconf]
          sumd+=sim[ind1][ind2]
          icount+=1
    meand=sumd/icount
#   calculate variance and sd
    var=0.0
    icount=0
    minvar=9999
    for iconf in range(len(clusterlist[i])):
        ind1=clusterlist[i][iconf]
        ivar=0.0
        for jconf in range(len(clusterlist[i])):
          ind2=clusterlist[i][jconf]
          ivar+=(sim[ind1][ind2]-meand)**2
        ivar=ivar/(len(clusterlist[i])-1)
        var+=ivar
        icount+=1
        if(ivar<minvar):
            minvar=ivar
            iselect=ind1
    rep_index.append(iselect)
    var=var/(icount)
    sd=np.sqrt(var)
    sdlist.append(sd)
  return sdlist,rep_index
#    print len(clusterlist[i]),meand,var,sd,iselect,Z[i,2]
#  print "clusters:", nl-elbow+2

def estimate_ncluster(dist,dcut):
  n=len(dist)
  if dcut>0.0 : 
    for i in range(n):
      if dist[i]>dcut : 
          nclust=n-i
          break

  else:
    b=[n-1,dist[n-1]-dist[0]]
    b=np.array(b)
    b=b/np.linalg.norm(b)
    dmax=0.0
    for i in range(n):
      p=[n-1-i,dist[n-1]-dist[i]]
      d=np.linalg.norm(p-np.dot(p,b)*b)
      if d>dmax :
         elbow=i
         dmax=d
    dcut=dist[elbow]*1.2
    print "estimated dcut=",dcut
    for j in range(n):
      if dist[j]>dcut : 
            nclust=n-j
            break
  return nclust


def prop_sd(Z,prop,verbose):
  n=len(prop)
  clusterlist=[]
  ncluster=0
  sdlist=[]
  if verbose : f=open('clusterlist.dat','w')
  for i in range(len(Z)):
    id1=int(Z[i,0])
    id2=int(Z[i,1])
    if((id1 < n) and (id2<n)):  # when two configurations are merged note their index
       clusterlist.append([id1,id2])
       ncluster+=1
    else:
      cl=[]
      icount=0
      if id1>=n:  # this means merging is happening with previously formed cluster
        icluster=int(id1)-n
        for x in clusterlist[icluster]: #we already have the list for the old cluster
          cl.append(x)
      else:cl.append(id1)
      if id2>=n: # same logic as before
        icluster=int(id2)-n
        for x in clusterlist[icluster]:
          cl.append(x)
      else:cl.append(id2)
      clusterlist.append(cl) # append the index of the members at this stage of clustering 
#   calculate mean dissimilarity of the cluster
    sumd=0.0
    icount=0
#   calculate variance and sd
    sd=np.std(prop[clusterlist[i]])
    if verbose: 
      f.write(" %s " %str(clusterlist[i]))
      f.write("\n")
    sdlist.append(sd)
  return sdlist
#    print len(clusterlist[i]),meand,var,sd,iselect,Z[i,2]
#  print "clusters:", nl-elbow+2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Computes KRR and analytics based on a kernel matrix and a property vector.""")

    parser.add_argument("sim", nargs=1, help="Kernel matrix")
    parser.add_argument("--mode", type=str, default="average", help="Linkage mode (e.g. --mode average/single/complete/median/centroid")
    parser.add_argument("--dcut", type=float, default='0', help="distance cutoff to cut the dendrogram. if dcut=0 then it is autamaticlly estimated")
    parser.add_argument("--prop", type=str, default='', help="property file")
    parser.add_argument("--plot",  action="store_true", help="Plot the dendrogram")
    parser.add_argument("--calc_sd",  action="store_true", help="calculate standard div of the dist and prop for all level of clustering")
    parser.add_argument("--project",  type=str,default='', help="Project configurations using Rect Dist Matrix file")
    parser.add_argument("--mathematica",  action="store_true", help="export the cluster object in Mathematica format")
    parser.add_argument("--verbose",  action="store_true", help="increase output informations. write multiple files")

    args = parser.parse_args()
    main(args.sim[0],args.dcut,mode=args.mode,proplist=args.prop,plot=args.plot,calc_sd=args.calc_sd,rect_matrixfile=args.project,mtcia=args.mathematica,verbose=args.verbose)

