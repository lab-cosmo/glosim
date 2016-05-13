"""
Compute correlation matrix between the clustering of 2 differents environment
"""
import argparse
import numpy as np
import sys
import scipy.cluster.hierarchy as sc
import itertools
from scipy.stats import kurtosis,skew
from scipy.stats.mstats import kurtosistest
from os.path import basename
import cluster
from collections import Counter
import matplotlib.pyplot as plt
import quippy

def getcorrofcluster(mols,cutoffdist,fnamexyz,clusterlist1,clusterlist2,ng1,ng2,zenv1,zenv2,nspiecies1,nspiecies2,selectgroupxyz2print):
	# zenv is atomic nb of selected atom and nenv is the number of 
	# such species in the similarity matrix

	nframe = len(mols)
	
	# initialize correlation matrix 
	corrmtx = np.zeros((ng1,ng2), dtype=int)  
	
	# give correlation data structured with specie of clusterlist1 
 	# as main element
 	corr1 = {x: [] for x in xrange(ng1)}
 	corr2 = {x: [] for x in xrange(ng2)}
 	
 	r1 = np.array([0,0,0],dtype=float)
 	r2 = np.array([0,0,0],dtype=float)

 	# count nb of atom in each clst group
 	atmcount = {x: -1 for x in xrange(ng1)}
 	# Print xyz of selected group
	if len(selectgroupxyz2print) > 0 :
	 	selectfname = 'selected'+str(selectgroupxyz2print[0])+'_'+str(selectgroupxyz2print[1])+'.env'+str(zenv1)+'_'+str(zenv2)+'.xyz'
		print 'Writing ', selectfname
	 	f = open(selectfname,'w')

	# Main loop over the frames 	
 	for iframe in xrange(nframe):
 		mol = mols[iframe]

	 	for atm1 in xrange(nspiecies1[iframe]): # mol.z is fortran array bound:(1:natm)

	 		# need to be done at each loop
	 		corr2 = {x: [] for x in xrange(ng2)}

	 		corr1[clusterlist1[iframe][atm1][0]].append([clusterlist1[iframe][atm1][1],clusterlist1[iframe][atm1][2],{zenv2 : corr2}])
	 		atmcount[clusterlist1[iframe][atm1][0]] += 1
	 		
	 		# position of the atom from clusterlist1
	 		r1[0] = mol[clusterlist1[iframe][atm1][2]].x
	 		r1[1] = mol[clusterlist1[iframe][atm1][2]].y
	 		r1[2] = mol[clusterlist1[iframe][atm1][2]].z
	 		# print r1
	 		# print 'Frame : ', iframe, ' and atm : ', atm1
	 		for atm2 in xrange(nspiecies2[iframe]):
	 			# position of the atom from clusterlist2
	 			# print nspiecies1[iframe]
	 			# print clusterlist2[iframe]
	 			# print clusterlist2[iframe][atm2]
	 			# print clusterlist2[iframe][atm2][2]
	 			r2[0] = mol[clusterlist2[iframe][atm2][2]].x
	 			r2[1] = mol[clusterlist2[iframe][atm2][2]].y
	 			r2[2] = mol[clusterlist2[iframe][atm2][2]].z
	 			
	 			dist = np.linalg.norm(r1-r2)
	 			# if atom is within cutoff distance then update correlation matrix
	 			if dist < cutoffdist: 
	 				corrmtx[clusterlist1[iframe][atm1][0],clusterlist2[iframe][atm2][0]] += 1 
	 				corr1[clusterlist1[iframe][atm1][0]][atmcount[clusterlist1[iframe][atm1][0]]][2][zenv2][clusterlist2[iframe][atm2][0]].append(clusterlist2[iframe][atm2][2])
	 				# Print xyz of selected group
					if len(selectgroupxyz2print) > 0 :
						if 	(int(clusterlist1[iframe][atm1][0]) == selectgroupxyz2print[0] and int(clusterlist2[iframe][atm2][0]) == selectgroupxyz2print[1]):
							# print 'TTTTTTTTTTTTTTTTTTT'
							tmppos=[]
							tmpname= []

							for atm3 in xrange(mol.n):
								# important otherwise r3 is not updated properly
								r3 = []
								r3.extend([mol[atm3].x, mol[atm3].y,mol[atm3].z])
								
								dist1 = np.linalg.norm(r1-r3)
					 			if dist1 < cutoffdist:
					 				#print r3
									tmppos.append(r3)
									tmpname.append(mol[atm3].symbol)
							f.write(str(len(tmppos))+'\n')
							f.write('Frame '+str(iframe)+' from '+fnamexyz+'\n')
							# print tmppos 
							for line in xrange(len(tmppos)):
							# 	f.write('{0:} \t {1:.8f} \t {2:.8f}\t {3:.8f} \n'.format(tmpname[line],tmppos[line][0],tmppos[line][1],tmppos[line][2]))
								f.write("%s %f %f %f \n" %(tmpname[line],tmppos[line][0],tmppos[line][1],tmppos[line][2]))
	if len(selectgroupxyz2print) > 0 :
		f.close()
	# corrkeys = corr1[0][0][2][zenv2].keys()
	# arr = np.zeros(len(corrkeys))
	# for i in range(0,len(corr1[0])):
	# 	for j in range(len(corrkeys)):
	# 		arr[j] += len(corr1[0][i][2][zenv2][corrkeys[j]])
	# 	print corr1[0][i][0], corr1[0][i][1]
	# 	print corr1[0][i][2][zenv2]
	# print corrmtx 
	# print arr
	return corrmtx, corr1 

def rmdummyfromsim(fnamexyz,distmatrix,zenv,nenv):
	# zenv is atomic nb of env and nenv is the number of 
 	# such species in the similarity matrix
	
	# print clist
	atomicmap = {1:"H",2:"He",6:"C",7:"N",8:"O",9:"F"}
	# atomicmap_inv = {"H":1,"He":2,"C":6,"N":7,"O":8,"F":9}
	# get the atom name and its number from filename
	symbenv = atomicmap[zenv]
	fxyz = open(fnamexyz,'r')

	dummylist = np.ones((len(distmatrix)),dtype=bool)

	atomnb = 0
	# list on the atom : (its cluster idx, its frame idx, id in frame ignoring other atoms, id in frame)
	for ind in xrange(len(distmatrix)):
		iat = np.fmod(ind,nenv) # ind starts at 0
		# iframe = (ind - iat) / nenv
   
		if iat==0: # if new frame 
			atomnb = int(fxyz.readline()) # get number of atoms in the frame
			fxyz.readline() # skip the comment line
			strt = 0
			lines = []
			for it in xrange(atomnb): # reads the full frame
				lines.append(fxyz.readline())

		for it in xrange(strt,atomnb):
			isdummy = False	
			if lines[it].find(symbenv)>=0: # find() returns -1 if does not find the atom name in line
				strt=it+1
				break

			
			if it >= nenv-1: # if the atom is a dummy then set its idx nb to -1
				strt = atomnb
				#print ind		
				isdummy = True
		if isdummy:
			dummylist[ind] = False 
	
	tmp = distmatrix[dummylist,:]
	newdistmtr = tmp[:,dummylist]
	
	fxyz.close()
	return newdistmtr, dummylist


def linkgroup2atmidx(mols,clist,zenv,nenv):
	# zenv is atomic nb of env and nenv is the number of 
 	# such species in the similarity matrix
	
	nframe = len(mols)
	# symbsim = atomicmap[zenv]

	# tmp = np.zeros(3,dtype=int)
	# clusterlist = np.zeros((len(clist),3),dtype=int)
	clusterlist = []
	clistcntr = 0
	nspecies = np.zeros(nframe,dtype=int)
	# list on the atom : (its cluster idx, its frame idx, id in frame ignoring other atoms, id in frame)
	for iframe in xrange(nframe):
		mol = mols[iframe]
		# nenv = 0
		natm = len(mol.z)
		clusterlist.append([])
		for it in xrange(1,natm+1): # mol.z is a fortran array (1:natm)
			if mol.z[it] == zenv:
				nspecies[iframe] += 1
				tmp1 = [int(clist[clistcntr]), iframe, it -1]
				clusterlist[iframe].append(tmp1)
				clistcntr += 1

	return clusterlist, nspecies 


def clusterdistmat(distmatrixfile,sim,dcut,mode='average',plot=False):
	# Compute the clusturing on dist^2 so that the average 
	# distance of a cluster with an other is the RMS distance
	sim2 = sim*sim
	Z = sc.linkage(sim2,mode)

	cdist = Z[:,2]

   	nclust = cluster.estimate_ncluster(cdist,dcut)

	clist = sc.fcluster(Z,nclust,criterion='maxclust')
	c_count = Counter(clist)
	nbclst = len(c_count)

	print "Number of clusters", nbclst 
	
	rep_ind = getrep_ind(sim2,clist,c_count)

	# Write the groupe indices and representatives
	filename=basename(distmatrixfile)+'-cluster.index'
	f=open(filename,"w")
	f.write(" # groupid representative \n ")
	for i in range(len(sim)):
		iselect=0
		if i in rep_ind: iselect=2
		f.write("%d   %d \n " %(clist[i]-1,  iselect)) 
	f.close()

   
	if plot: 
		filename=basename(distmatrixfile)+'-dendogram.eps'
		plotdendro(Z,nclust,filename,rep_ind)
	c_list = np.zeros(len(sim))

	# Change cluster groups numbering to (0:n-1)
	for i in range(len(sim)):
		c_list[i] = int(clist[i]-1)

	return c_list

# Determine the representative element of each cluster group
def getrep_ind(sim2,clist,c_count):
	rep_ind=[]
	structurelist=[]
	for iclust in range(1,len(c_count)+1):  #calculate mean dissimilary and pick representative structure for each cluster
		indices = [i for i, x in enumerate(clist) if x == iclust] #indices for cluster i
		nconf=len(indices)
		structurelist.append(indices)
		sumd=0.0

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
	return rep_ind
# Plot and print the dendogram 
def plotdendro(Z,ncluster,filename,rep_ind):
	plt.figure(figsize=(10, 15))
	plt.title('Hierarchical Clustering Dendrogram')
	plt.xlabel('sample index')
	plt.ylabel('distance')

	d = sc.dendrogram(Z,truncate_mode='lastp', p=ncluster,orientation='right',leaf_rotation=90.,leaf_font_size=20.,show_contracted=False)
	
	coord=[]
	for i in range(len(d['icoord'])):
		if d['dcoord'][i][0]==0.0 :
			coord.append(d['icoord'][i][0])
	for i in range(len(d['icoord'])):
		if d['dcoord'][i][3]==0.0 :
			coord.append(d['icoord'][i][3])

	plt.savefig(filename, dpi=100, facecolor='w', edgecolor='w',
        orientation='portrait', papertype='letter', format=None,
        transparent=True, bbox_inches=None, pad_inches=0.1,
        frameon=None)