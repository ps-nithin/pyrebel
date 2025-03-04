# Copyright (C) 2024-2025 Nithin PS.
# This file is part of Pyrebel.
#
# Pyrebel is free software: you can redistribute it and/or modify it under the terms of 
# the GNU General Public License as published by the Free Software Foundation, either 
# version 3 of the License, or (at your option) any later version.
#
# Pyrebel is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
# PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Pyrebel.
# If not, see <https://www.gnu.org/licenses/>.
#

from numba import cuda
import math
import numpy as np
from math import sqrt
import os
os.environ['NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS']="0"
from pyrebel.getnonzeros import *

@cuda.jit
def fence_image(img_array,img_fenced_d):
    r,c=cuda.grid(2)
    if r>=0 and r<img_array.shape[0] and c>=0 and c<img_array.shape[1]:
        if r==0 or c==0 or r==img_array.shape[0]-1 or c==img_array.shape[1]-1:
            img_fenced_d[r][c]=500
@cuda.jit
def scale_img_cuda(img,img_scaled):
    r,c=cuda.grid(2)
    if r<img.shape[0] and c<img.shape[1]:
        img_scaled[r*3][c*3]=img[r][c]
        img_scaled[r*3][c*3+1]=img[r][c]
        img_scaled[r*3][c*3+2]=img[r][c]
        img_scaled[r*3+1][c*3]=img[r][c]
        img_scaled[r*3+1][c*3+1]=img[r][c]
        img_scaled[r*3+1][c*3+2]=img[r][c]
        img_scaled[r*3+2][c*3]=img[r][c]
        img_scaled[r*3+2][c*3+1]=img[r][c]
        img_scaled[r*3+2][c*3+2]=img[r][c]

@cuda.jit
def read_bound_cuda(img,img_boundary_d):
    """ blob_dict={color: [[pixels],[pixels]]"""
    r,c=cuda.grid(2)
    threshold=0
    if r>0 and r<img.shape[0]-1 and c>0 and c<img.shape[1]-1:
        if abs(img[r][c]-img[r][c+1])>threshold: # left ro right
            img_boundary_d[r][c]=img[r][c]
            img_boundary_d[r][c+1]=img[r][c+1]
        
        if abs(img[r][c]-img[r+1][c])>threshold: # top to bottom
            img_boundary_d[r][c]=img[r][c]
            img_boundary_d[r+1][c]=img[r+1][c]
        
        if abs(img[r][c]-img[r+1][c+1])>threshold: # diagonal
            img_boundary_d[r][c]=img[r][c]
            img_boundary_d[r+1][c+1]=img[r+1][c+1]

        if abs(img[r+1][c]-img[r][c+1])>threshold: # diagonal
            img_boundary_d[r+1][c]=img[r+1][c]
            img_boundary_d[r][c+1]=img[r][c+1]    

@cuda.jit
def get_bound_cuda2(tmp_img,bound_len_low,bound_len_high,seed_map_d,bound_info):
    r,c=cuda.grid(2)
    # last=0,1,2,3 for n,e,s,w respectively
    if r>0 and r<tmp_img.shape[0]-1 and r%3==0 and c>0 and c<tmp_img.shape[1]-1 and c%3==0 and tmp_img[r][c]!=500 and tmp_img[r][c-1]!=tmp_img[r][c+1] and tmp_img[r-1][c]!=tmp_img[r+1][c]:
        y=r
        x=c
        color=tmp_img[r][c]
        n=1
        cur_i=r*tmp_img.shape[1]+c
        min_i=cur_i
        last=-1
        if tmp_img[r-1][c]==color:
            r-=1
            last=2
        elif tmp_img[r][c+1]==color:
            c+=1
            last=3
        elif tmp_img[r+1][c]==color:
            r+=1
            last=0
        elif tmp_img[r][c-1]==color:
            c-=1
            last=1
        while 1:
            if r==y and c==x:
                break
            n+=1
            cur_i=r*tmp_img.shape[1]+c
            if cur_i<min_i:
                min_i=cur_i
            if tmp_img[r-1][c]==color and last!=0:
                r-=1
                last=2
            elif tmp_img[r][c+1]==color and last!=1:
                c+=1
                last=3
            elif tmp_img[r+1][c]==color and last!=2:
                r+=1
                last=0
            elif tmp_img[r][c-1]==color and last!=3:
                c-=1
                last=1
        cuda.syncthreads()
        if n>bound_len_low and n<bound_len_high:
            bound_info[min_i][0]=min_i
            bound_info[min_i][1]=n
        seed_map_d[cur_i]=min_i
        #cuda.atomic.max(bound_max_d[min_i],0,d_tmp_max)

@cuda.jit
def get_bound_data_init(nz_a,nz_s,tmp_img,bound_data_d):
    ci=cuda.grid(1)
    if ci<nz_a.shape[0]:
        index=nz_a[ci]
        y=int(index/tmp_img.shape[1])
        x=index%tmp_img.shape[1]
        r=y
        c=x
        color=tmp_img[r][c]
        n=nz_s[ci]
        bound_data_d[n]=r*tmp_img.shape[1]+c
        last=-1
        if tmp_img[r-1][c]==color:
            r-=1
            last=2
        elif tmp_img[r][c+1]==color:
            c+=1
            last=3
        elif tmp_img[r+1][c]==color:
            r+=1
            last=0
        elif tmp_img[r][c-1]==color:
            c-=1
            last=1
        while 1:
            if r==y and c==x:
                break
            n+=1
            bound_data_d[n]=r*tmp_img.shape[1]+c

            if tmp_img[r-1][c]==color and last!=0:
                r-=1
                last=2
            elif tmp_img[r][c+1]==color and last!=1:
                c+=1
                last=3
            elif tmp_img[r+1][c]==color and last!=2:
                r+=1
                last=0
            elif tmp_img[r][c-1]==color and last!=3:
                c-=1
                last=1

@cuda.jit
def get_dist_data_init(bound_data_d,tmp_img,dist_data_d):
    ci=cuda.grid(1)
    if ci<len(bound_data_d):
        index=bound_data_d[ci]
        y=int(index/tmp_img.shape[1])
        x=index%tmp_img.shape[1]
        r=y
        c=x
        max_r=y
        max_c=x
        d_max=0.0
        color=tmp_img[r][c]
        last=-1
        if tmp_img[r-1][c]==color:
            r-=1
            last=2
        elif tmp_img[r][c+1]==color:
            c+=1
            last=3
        elif tmp_img[r+1][c]==color:
            r+=1
            last=0
        elif tmp_img[r][c-1]==color:
            c-=1
            last=1
        while 1:
            if r==y and c==x:
                break
            d_cur=sqrt(float(pow(r-y,2)+pow(c-x,2)))
            if d_cur>d_max:
                d_max=d_cur
                max_r=r
                max_c=c
            if tmp_img[r-1][c]==color and last!=0:
                r-=1
                last=2
            elif tmp_img[r][c+1]==color and last!=1:
                c+=1
                last=3
            elif tmp_img[r+1][c]==color and last!=2:
                r+=1
                last=0
            elif tmp_img[r][c-1]==color and last!=3:
                c-=1
                last=1
        dist_data_d[ci][0]=d_max
        dist_data_d[ci][1]=max_r*tmp_img.shape[1]+max_c
        

@cuda.jit
def get_max_dist(nz_s_cum_d,nz_s_d,bound_data_d,dist_data_d,max_dist_d):
    ci=cuda.grid(1)
    if ci<len(nz_s_d):
        n=nz_s_cum_d[ci]
        s=0
        d_max=dist_data_d[n][0]
        d_max_i=n
        while 1:
            s+=1
            if dist_data_d[n][0]>d_max:
                d_max=dist_data_d[n][0]
                d_max_i=n
            if s==nz_s_d[ci]:
                break
            n+=1

        max_dist_d[ci][0]=bound_data_d[d_max_i]
        max_dist_d[ci][1]=int(dist_data_d[d_max_i][1])
        
@cuda.jit
def get_bound_data_order(nz_a_max_dist,nz_si_cum_d,tmp_img,init_bound_abstract,bound_data_order_d,bound_threshold_d,bound_mark_d,ba_size_d,threshold_in):
    ci=cuda.grid(1)
    if ci<len(nz_a_max_dist):
        index=nz_a_max_dist[ci][0]
        index2=nz_a_max_dist[ci][1]
        y=int(index/tmp_img.shape[1])
        x=index%tmp_img.shape[1]
        y2=int(index2/tmp_img.shape[1])
        x2=index2%tmp_img.shape[1]
        threshold_ratio=threshold_in
        threshold=sqrt(float(pow(y2-y,2)+pow(x2-x,2)))/threshold_ratio
        if threshold<5:
            threshold=5
        r=y
        c=x
        color=tmp_img[r][c]
        n=nz_si_cum_d[ci]
        init_n=ci
        init_bound_abstract[n]=n+1
        bound_threshold_d[n]=threshold
        bound_mark_d[n]=init_n
        cuda.atomic.add(ba_size_d,init_n,1)
        bound_data_order_d[n]=r*tmp_img.shape[1]+c
        last=-1
        if tmp_img[r-1][c]==color:
            r-=1
            last=2
        elif tmp_img[r][c+1]==color:
            c+=1
            last=3
        elif tmp_img[r+1][c]==color:
            r+=1
            last=0
        elif tmp_img[r][c-1]==color:
            c-=1
            last=1
        while 1:
            if r==y and c==x:
                bound_data_order_d[n+1]=y*tmp_img.shape[1]+x
                init_bound_abstract[n+1]=n+2
                bound_threshold_d[n+1]=threshold
                bound_mark_d[n+1]=init_n
                cuda.atomic.add(ba_size_d,init_n,1)
                break
            n+=1
            bound_data_order_d[n]=r*tmp_img.shape[1]+c
            bound_threshold_d[n]=threshold
            bound_mark_d[n]=init_n

            if y2==r and x2==c:
                init_bound_abstract[n]=n+1
                cuda.atomic.add(ba_size_d,init_n,1)
                
            if tmp_img[r-1][c]==color and last!=0:
                r-=1
                last=2
            elif tmp_img[r][c+1]==color and last!=1:
                c+=1
                last=3
            elif tmp_img[r+1][c]==color and last!=2:
                r+=1
                last=0
            elif tmp_img[r][c-1]==color and last!=3:
                c-=1
                last=1

@cuda.jit
def increment_by_one(array_d):
    ci=cuda.grid(1)
    if ci<len(array_d):
        array_d[ci]+=1
        cuda.syncthreads()

@cuda.jit
def decrement_by_one(array_d):
    ci=cuda.grid(1)
    if ci<len(array_d):
        array_d[ci]-=1
        cuda.syncthreads()
                     
class Preprocess:
    def __init__(self,img_array):
        self.img_array=img_array
        self.bound_len_low=64
        self.bound_len_high=img_array.shape[0]*img_array.shape[1]*9
        self.nz_s=[]
        self.nz_a=[]
        self.bound_data_ordered_h=[]
        self.bound_abstract_h=[]
        self.img_scaled_h=[]
        self.bound_mark_h=[]
        
    def preprocess_image(self):
        img_array_d=cuda.to_device(self.img_array)
        threadsperblock=(16,16)
        blockspergrid_x=math.ceil(self.img_array.shape[0]/threadsperblock[0])
        blockspergrid_y=math.ceil(self.img_array.shape[1]/threadsperblock[1])
        blockspergrid=(blockspergrid_x,blockspergrid_y)
        img_fenced_d=img_array_d
        fence_image[blockspergrid,threadsperblock](img_array_d,img_fenced_d)
        cuda.synchronize()
        scaled_shape=np.array([self.img_array.shape[0]*3,self.img_array.shape[1]*3])
        scaled_shape_d=cuda.to_device(scaled_shape)
        img_scaled_d=cuda.device_array(scaled_shape,dtype=np.int32)
        scale_img_cuda[blockspergrid,threadsperblock](img_fenced_d,img_scaled_d)
        cuda.synchronize()
        self.img_scaled_h=img_scaled_d.copy_to_host()
        
        threadsperblock=(16,16)
        blockspergrid_x=math.ceil(scaled_shape[0]/threadsperblock[0])
        blockspergrid_y=math.ceil(scaled_shape[1]/threadsperblock[1])
        blockspergrid=(blockspergrid_x,blockspergrid_y)
        img_boundary=np.full(scaled_shape,500,dtype=np.int32)
        img_boundary_d=cuda.to_device(img_boundary)
        read_bound_cuda[blockspergrid,threadsperblock](img_scaled_d,img_boundary_d)
        cuda.synchronize()
        bound_info=np.zeros([scaled_shape[0]*scaled_shape[1],2],dtype=np.int32)
        bound_info_d=cuda.to_device(bound_info)
        seed_map=np.zeros(scaled_shape[0]*scaled_shape[1],dtype=np.int32)
        seed_map_d=cuda.to_device(seed_map)
        threadsperblock=(16,16)
        blockspergrid_x=math.ceil(scaled_shape[0]/threadsperblock[0])
        blockspergrid_y=math.ceil(scaled_shape[1]/threadsperblock[1])
        blockspergrid=(blockspergrid_x,blockspergrid_y)
        get_bound_cuda2[blockspergrid,threadsperblock](img_boundary_d,self.bound_len_low,self.bound_len_high,seed_map_d,bound_info_d)
        cuda.synchronize()
        
        binfo=bound_info_d.copy_to_host()
        a=binfo.transpose()[0]
        s=binfo.transpose()[1]
        self.nz_a=get_non_zeros(a)
        self.nz_s=get_non_zeros(s)

        
        #print("len(nz_s)=",len(self.nz_s))
        #nz=np.column_stack((nz_a,nz_s))
        #nz_sort=nz[nz[:,1].argsort()]
        nz_s_cum_=np.cumsum(self.nz_s)
        nz_s_cum=np.delete(np.insert(nz_s_cum_,0,0),-1)
        nz_s_cum_d=cuda.to_device(nz_s_cum)
        nz_a_d=cuda.to_device(self.nz_a)
        nz_s_d=cuda.to_device(self.nz_s)

        """
        binfo_sort=np.argsort(nz_a)
        binfo_map=binfo_sort[np.searchsorted(nz_a,a,sorter=binfo_sort)]
        binfo_map_d=cuda.to_device(binfo_map)
 
        max_i=np.argmax(nz_s)
        width=nz_s[max_i]
        neighbor_data=np.zeros([len(nz_a),width],dtype=np.int32)
        neighbor_data_d=cuda.to_device(neighbor_data)
 
        get_neighbor_data_init2[blockspergrid,threadsperblock](img_boundary_d,binfo_map_d,seed_map_d,30,neighbor_data_d)
        cuda.synchronize()
        neighbor_data_h=neighbor_data_d.copy_to_host()
        
        out_image=np.zeros(scaled_shape,dtype=np.int32)
        out_image_d=cuda.to_device(out_image)
 
        
        img_boundary_h=img_boundary_d.copy_to_host()       
        draw_pixels_cuda(neighbor_data_d[3],255,img_boundary_d)
        # neighbor end
        """
        
        nz_si_d=cuda.to_device(self.nz_s)
        increment_by_one[len(self.nz_s),1](nz_si_d)
        nz_si=nz_si_d.copy_to_host()
        nz_si_cum_=np.cumsum(nz_si)
        nz_si_cum=np.delete(np.insert(nz_si_cum_,0,0),-1)
        nz_si_cum_d=cuda.to_device(nz_si_cum)

        bound_data_d=cuda.device_array([nz_s_cum_[-1]],dtype=np.int32)
        get_bound_data_init[math.ceil(len(self.nz_a)/256),256](nz_a_d,nz_s_cum_d,img_boundary_d,bound_data_d)
        cuda.synchronize()

        dist_data_d=cuda.device_array([nz_s_cum_[-1],2],dtype=np.float64)
        get_dist_data_init[math.ceil(nz_s_cum_[-1]/256),256](bound_data_d,img_boundary_d,dist_data_d)
        cuda.synchronize()
        
        max_dist_d=cuda.device_array([len(self.nz_s),2],dtype=np.int32)
        get_max_dist[math.ceil(len(self.nz_s)/1),1](nz_s_cum_d,nz_s_d,bound_data_d,dist_data_d,max_dist_d)
        cuda.synchronize()

        bound_data_ordered_d=cuda.device_array([nz_si_cum_[-1]],dtype=np.int32)
        bound_abstract=np.zeros([nz_si_cum_[-1]],dtype=np.int32)
        bound_abstract_d=cuda.to_device(bound_abstract)
        bound_threshold=np.zeros([nz_si_cum_[-1]],dtype=np.float64)
        bound_mark_d=cuda.device_array([nz_si_cum_[-1]],dtype=np.int32)
        bound_threshold_d=cuda.to_device(bound_threshold)
        ba_size=np.zeros([nz_si_cum_[-1]],dtype=np.int32)
        ba_size_d=cuda.to_device(ba_size)
        get_bound_data_order[math.ceil(len(self.nz_a)/256),256](max_dist_d,nz_si_cum_d,img_boundary_d,bound_abstract_d,bound_data_ordered_d,bound_threshold_d,bound_mark_d,ba_size_d,5)
        cuda.synchronize()
        self.bound_data_ordered_h=bound_data_ordered_d.copy_to_host()
        self.bound_abstract_h=bound_abstract_d.copy_to_host()
        self.bound_mark_h=bound_mark_d.copy_to_host()    
        
    def set_bound_size(self,min_size=False,max_size=False):
        if min_size:
            self.bound_len_low=min_size
        if max_size:
            self.bound_len_high=max_size
    
    def get_bound_size(self):
        return self.nz_s
    def get_bound_seed(self):
        return self.nz_a
        
    def get_image_scaled(self):
        return self.img_scaled_h
        
    def get_bound_data(self):
        return self.bound_data_ordered_h
    
    def get_bound_mark(self):
        return self.bound_mark_h
    
    def get_init_abstract(self):
        return self.bound_abstract_h
    
