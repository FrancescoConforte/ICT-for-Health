# -*- coding: utf-8 -*-

import numpy   as np
import nibabel as nib # to read NII files
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from itertools import product

#%% methods
def read_nii(filepath):
    '''
    Reads .nii file and returns pixel array
    '''
    ct_scan = nib.load(filepath)
    array   = ct_scan.get_fdata()
    array   = np.rot90(np.array(array))
    return(array)

def plotSample(array_list, color_map = 'nipy_spectral'):
    '''
    Plots a slice with all available annotations
    '''
    plt.figure(figsize=(18,15))

    plt.subplot(1,4,1)
    plt.imshow(array_list[0], cmap='bone')
    plt.title('Original Image')

    plt.subplot(1,4,2)
    plt.imshow(array_list[0], cmap='bone')
    plt.imshow(array_list[1], alpha=0.5, cmap=color_map)
    plt.title('Lung Mask')

    plt.subplot(1,4,3)
    plt.imshow(array_list[0], cmap='bone')
    plt.imshow(array_list[2], alpha=0.5, cmap=color_map)
    plt.title('Infection Mask')

    plt.subplot(1,4,4)
    plt.imshow(array_list[0], cmap='bone')
    plt.imshow(array_list[3], alpha=0.5, cmap=color_map)
    plt.title('Lung and Infection Mask')

    plt.show()
    

def filterImage(D,NN):
    """D = image (matrix) to be filtered, Nr rows, N columns, scalar values (no RGB color image)
    The image is filtered using a square kernel/impulse response with side 2*NN+1"""
    E=D.copy()
    E[np.isnan(E)]=0
    Df=E*0
    Nr,Nc=D.shape
    rang=np.arange(-NN,NN+1)
    square=np.array([x for x in product(rang, rang)])
    #square=np.array([[1,1],[1,0],[1,-1],[0,1],[0,0],[0,-1],[-1,1],[-1,0],[-1,-1]])
    for kr in range(NN,Nr-NN):
        for kc in range(NN,Nc-NN):
            ir=kr+square[:,0]
            ic=kc+square[:,1]
            Df[kr,kc]=np.sum(E[ir,ic])# Df will have higher values where ones are close to each other in D
    return Df/len(square)

def useDBSCAN(D,z,epsv,min_samplesv):
    """D is the image to process, z is the list of image coordinates to be
    clustered"""
    Nr,Nc=D.shape
    clusters =DBSCAN(eps=epsv,min_samples=min_samplesv,metric='euclidean').fit(z)
    Nclust_DBSCAN=np.max(clusters.labels_)+1
    
    Npoints_per_cluster=np.zeros((Nclust_DBSCAN,),dtype=int)
    for k in range(np.max(clusters.labels_)):
        ii=(clusters.labels_==k)
        Npoints_per_cluster[k]=np.sum(ii)
    ii=np.argsort(-Npoints_per_cluster)# from the most to the less populated clusters
    Npoints_per_cluster=Npoints_per_cluster[ii]
    C=np.zeros((Nr,Nc,Nclust_DBSCAN))*np.nan # one image for each cluster
    info=np.zeros((Nclust_DBSCAN,5),dtype=float)
    for k in range(Nclust_DBSCAN):
        i1=ii[k] 
        index=(clusters.labels_==i1)
        jj=z[index,:] # coordinates of cluster k
        C[jj[:,0],jj[:,1],k]=1 # Ndarray k stores cluster k
        a=np.mean(jj,axis=0).tolist()
        b=np.var(jj,axis=0).tolist()
        info[k,0:2]=a #  store coordinates of the centroid
        info[k,2:4]=b # store variance
        info[k,4]=Npoints_per_cluster[k] # store points in cluster
    return C,info,clusters
    
#%% main part
        
# Read sample
plt.close('all')    
plotFlag=True

fold1='./ct_scans'
fold2='./lung_mask'
fold3='./infection_mask'
fold4='./lung_and_infection_mask'
f1='/coronacases_org_001.nii'
f2='/coronacases_001.nii'
sample_ct   = read_nii(fold1+f1+f1)
sample_lung = read_nii(fold2+f2+f2)
sample_infe = read_nii(fold3+f2+f2)
sample_all  = read_nii(fold4+f2+f2)

Nr,Nc,Nimages=sample_ct.shape# Nr=512,Nc=512,Nimages=301
#%% Examine one slice of a ct scan and its annotations
index=126 
sct=sample_ct[...,index]
sl=sample_lung[...,index]
si=sample_infe[...,index]
sa=sample_all[...,index]
plotSample([sct,sl,si,sa])


a=np.histogram(sct,200,density=True)
if plotFlag:
    plt.figure()
    plt.plot(a[1][0:200],a[0])
    plt.title('Histogram of CT values in slice '+str(index))
    plt.grid()
    plt.xlabel('value')
#%% Use Kmeans to perform color quantization of the image
Ncluster=5
kmeans = KMeans(n_clusters=Ncluster,random_state=0)# instantiate Kmeans
A=sct.reshape(-1,1)# Ndarray, Nr*Nc rows, 1 column
kmeans.fit(A)# run Kmeans on A
kmeans_centroids=kmeans.cluster_centers_.flatten() #centroids/quantized colors (which are the 5 representative colors)
for k in range(Ncluster):
    ind=(kmeans.labels_==k)# indexes for which the label is equal to k
    A[ind]=kmeans_centroids[k]# set the quantized color
sctq=A.reshape(Nr,Nc)# quantized image
vm=sct.min()
vM=sct.max()

fig, (ax1,ax2) = plt.subplots(1,2)
ax1.imshow(sct, cmap='bone')
ax1.set_title('Original image')
ax2.imshow(sctq,vmin=vm,vmax=vM, cmap='bone')
ax2.set_title('Quantized image')

ifind=1# second darkest color
D=A*np.nan
ii=kmeans_centroids.argsort()# sort centroids from lowest to highest
ind_clust=ii[ifind]# get the index of the desired cluster 
ind=(kmeans.labels_==ind_clust)# get the indexes of the pixels having the desired color
D[ind]=1# set the corresponding values of D  to 1
D=D.reshape(Nr,Nc)# make D an image/matrix through reshaping
plt.figure()
plt.imshow(D)
plt.title('Image used to identify lungs')
#%% DBSCAN to find the lungs in the image
C,centroids,clust=useDBSCAN(D,np.argwhere(D==1),2,5)
# we want left lung first. If the images are already ordered
# then the center along the y-axis (horizontal axis) of C[:,:,0] is smaller
if centroids[1,1]<centroids[0,1]:# swap the two subimages
    print('swap')
    tmp = C[:,:,0]*1
    C[:,:,0] = C[:,:,1]*1
    C[:,:,1] = tmp
    tmp=centroids[0,:]*1
    centroids[0,:]=centroids[1,:]*1
    centroids[1,:]=tmp
LLung = C[:,:,0].copy()  # left lung
RLung = C[:,:,1].copy()  # right lung

fig, (ax1,ax2) = plt.subplots(1,2)
ax1.imshow(LLung)
ax1.set_title('Left lung mask - initial')
ax2.imshow(RLung)
ax2.set_title('Right lung mask - initial')

#%% generate a new image with the two darkest colors of the color-quantized image
D=A*np.nan
ii=kmeans_centroids.argsort()# sort centroids from lowest to highest
ind=(kmeans.labels_==ii[0])# get the indexes of the pixels with the darkest color
D[ind]=1# set the corresponding values of D  to 1
ind=(kmeans.labels_==ii[1])# get the indexes of the pixels with the 2nd darkest  color
D[ind]=1# set the corresponding values of D  to 1
D=D.reshape(Nr,Nc)# make D an image/matrix through reshaping

C,centers2,clust=useDBSCAN(D,np.argwhere(D==1),2,5)
ind=np.argwhere(centers2[:,4]<1000) # remove small clusters
centers2=np.delete(centers2,ind,axis=0)
distL=np.sum((centroids[0,0:2]-centers2[:,0:2])**2,axis=1)    
distR=np.sum((centroids[1,0:2]-centers2[:,0:2])**2,axis=1)    
iL=distL.argmin()
iR=distR.argmin() 
LLungMask=C[:,:,iL].copy()
RLungMask=C[:,:,iR].copy()
fig, (ax1,ax2) = plt.subplots(1,2)
fig.tight_layout()
ax1.imshow(LLungMask)
ax1.set_title('Left lung mask - improvement')
ax2.imshow(RLungMask)
ax2.set_title('Right lung mask - improvement')

#%% Final lung masks

C,centers3,clust=useDBSCAN(LLungMask,np.argwhere(np.isnan(LLungMask)),1,5)
LLungMask=np.ones((Nr,Nc))
LLungMask[C[:,:,0]==1]=np.nan
C,centers3,clust=useDBSCAN(RLungMask,np.argwhere(np.isnan(RLungMask)),1,5)
RLungMask=np.ones((Nr,Nc))
RLungMask[C[:,:,0]==1]=np.nan

fig, (ax1,ax2) = plt.subplots(1,2)
ax1.imshow(LLungMask)
ax1.set_title('Left lung mask')
ax2.imshow(RLungMask)
ax2.set_title('Right lung mask')


fig, (ax1,ax2) = plt.subplots(1,2)
ax1.imshow(LLungMask*sct,vmin=vm,vmax=vM, cmap='bone')
ax1.set_title('Left lung')
ax2.imshow(RLungMask*sct,vmin=vm,vmax=vM, cmap='bone')
ax2.set_title('Right lung')

fig, (ax1,ax2) = plt.subplots(1,2)
ax1.imshow(LLungMask*sct)
ax1.set_title('Left lung')
ax2.imshow(RLungMask*sct)
ax2.set_title('Right lung')

#
#%% Quantize the colors of the lungs
LLungMask[np.isnan(LLungMask)]=0 #it is necessary to substitute nan with a number (0)
RLungMask[np.isnan(RLungMask)]=0
LungsMask=LLungMask+RLungMask #overall mask

B=LungsMask*sct #original image of lungs
inf_mask=1*(B>-650)&(B<-300)  #selected color interval 
InfectionMask=filterImage(inf_mask,1) #filter
InfectionMask=1.0*(InfectionMask>0.5)
InfectionMask[InfectionMask==0]=np.nan
plt.figure()
plt.imshow(InfectionMask)
plt.title('infection mask')

#final plot
color_map = 'spring'
plt.figure()
plt.imshow(sct,alpha=0.8,vmin=vm,vmax=vM, cmap='bone')
plt.imshow(InfectionMask*256,alpha=1,vmin=0,vmax=256, cmap=color_map)
plt.title('Original image with ground glass opacities in yellow')


#%% Features to understand how much of the lung is infected: PERCENTAGE + A SCORE
# Each of the five lung lobes was scored on a scale from 0 to 5, with:
    # 0, indicating no involvement;
    # 1, less than 5% involvement;
    # 2, 5%–25% involvement;
    # 3, 26%–49% involvement;
    # 4, 50%–75% involvement; 
    # 5, more than 75% involvement.
    
# The total CT score was the sum of the individual lobar
# scores and ranged from 0 (no involvement) to 25 (maximum involvement).

# To get a percentage of infected lung, number of pixels of left/right infection 
# mask equal to one and the ones of the total mask of 
# left/right lung always equal to one are counted and a percentage is done. 



# Find left/right infection mask separately
LInfectionMask=InfectionMask.copy() #save a copy of infection mask
LInfectionMask[:,250:]=np.nan #obtain left infection mask

RInfectionMask=InfectionMask.copy()
RInfectionMask[:,0:250]=np.nan #obtain right infection mask

fig, (ax1,ax2) = plt.subplots(1,2)
ax1.imshow(LInfectionMask)
ax1.set_title('Left infection mask')
ax2.imshow(RInfectionMask)
ax2.set_title('Right infection mask')


# Compute percentages

#left lung
n_pix_one_LL=np.count_nonzero(LLungMask) #pixels equal to one of entire 

temp=LInfectionMask.copy()
temp[np.isnan(LInfectionMask)]=0
n_pix_one_LI=np.count_nonzero(temp) #pixels equal to one of infection mask

L_percentage=(100*n_pix_one_LI)/n_pix_one_LL #percentage of infected area for left lung 

# score assignement
if L_percentage==0:
    score=0
elif L_percentage>0 and L_percentage<5:
    score=1
elif L_percentage>=5 and L_percentage<=25:
    score=2
elif L_percentage>=26 and L_percentage<=49:
    score=3
elif L_percentage>=50 and L_percentage<=75:
    score=4
elif L_percentage>75:
    score=5
    
print('For the left lung the percentage of infected lung is '+str(L_percentage)+'%')
print('For this reason the score is '+str(score))
    
#right lung
n_pix_one_RL=np.count_nonzero(RLungMask) #pixels equal to one of entire 

temp=RInfectionMask.copy()
temp[np.isnan(RInfectionMask)]=0
n_pix_one_RI=np.count_nonzero(temp) #pixels equal to one of infection mask

R_percentage=(100*n_pix_one_RI)/n_pix_one_RL #percentage of infected area for right lung 

# score assignement
if R_percentage==0:
    score=0
elif R_percentage>0 and R_percentage<5:
    score=1
elif R_percentage>=5 and R_percentage<=25:
    score=2
elif R_percentage>=25 and R_percentage<=49:
    score=3
elif R_percentage>=50 and R_percentage<=75:
    score=4
elif R_percentage>75:
    score=5

print('For the right lung the percentage of infected lung is '+str(R_percentage)+'%')
print('For this reason the score is '+str(score))






