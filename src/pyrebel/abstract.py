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

from pyrebel.getnonzeros import *
import numpy as np
from numba import cuda
import cmath,math

@cuda.jit
def find_ba_max_pd(nz_ba_d,nz_ba_size_d,bound_data_ordered_d,ba_max_pd_d,scaled_shape):
    ci=cuda.grid(1)
    if ci<len(nz_ba_d)-1:
        if nz_ba_d[ci]+1==nz_ba_d[ci+1]:
            return
        a=bound_data_ordered_d[nz_ba_d[ci]-1]
        b=bound_data_ordered_d[nz_ba_d[ci+1]-1]
        a0=int(a/scaled_shape[1])
        a1=a%scaled_shape[1]
        b0=int(b/scaled_shape[1])
        b1=b%scaled_shape[1]
        #threshold=bound_threshold_d[nz_ba_d[ci]]
        #threshold=cmath.sqrt(float(pow(b0-a0,2)+pow(b1-a1,2))).real/8
        n=nz_ba_d[ci]
        i=0
        pd_max=0.0
        pd_max_i=n
        while 1:
            if n==nz_ba_d[ci+1]:
                break
            c=bound_data_ordered_d[n]
            c0=int(c/scaled_shape[1])
            c1=c%scaled_shape[1]
            pd=abs((a1-b1)*(a0-c0)-(a0-b0)*(a1-c1))/cmath.sqrt(pow(a1-b1,2)+pow(a0-b0,2)).real

            if pd>pd_max:
                pd_max=pd
                pd_max_i=n+1
            n+=1
    ba_max_pd_d[ci][0]=pd_max
    ba_max_pd_d[ci][1]=pd_max_i
    """
    if pd_max>threshold:
        bound_abstract_d[pd_max_i]=pd_max_i
        seed_=bound_mark_d[nz_ba_d[ci]-1]
        #ba_size_d[seed_]+=1
        cuda.atomic.add(ba_size_d,seed_,1)
    """

@cuda.jit
def find_next_ba(ba_max_pd_d,nz_ba_size_d,nz_ba_size_cum_d,bound_abstract_d,ba_threshold,pd):
    ci=cuda.grid(1)
    if ci<len(nz_ba_size_d):
        n=nz_ba_size_cum_d[ci]
        s=1
        d_max=0.0
        d_max_i=n
        while 1:
            if ba_max_pd_d[n][0]>d_max:
                d_max=ba_max_pd_d[n][0]
                d_max_i=int(ba_max_pd_d[n][1])
            if s==nz_ba_size_d[ci]-1:
                break
            s+=1
            n+=1
    cuda.syncthreads()
    if d_max>ba_threshold:
        bound_abstract_d[d_max_i]=d_max_i
        nz_ba_size_d[ci]+=1
        pd[0]=d_max

@cuda.jit
def find_next_ba_all(ba_max_pd_d,nz_ba_size_d,nz_ba_size_cum_d,bound_abstract_d,ba_threshold):
    ci=cuda.grid(1)
    if ci<len(nz_ba_size_d):
        n=nz_ba_size_cum_d[ci]
        s=1
        ba_added=0
        #d_max=0.0
        #d_max_i=n
        while 1:
            if ba_max_pd_d[n][0]>ba_threshold:
                #d_max=ba_max_pd_d[n][0]
                d_max_i=int(ba_max_pd_d[n][1])
                bound_abstract_d[d_max_i]=d_max_i
                ba_added+=1
            if s==nz_ba_size_d[ci]-1:
                break
            s+=1
            n+=1
    cuda.syncthreads()
    cuda.atomic.add(nz_ba_size_d,ci,ba_added)
    #if d_max>ba_threshold:
    #    bound_abstract_d[d_max_i]=d_max_i
    #    nz_ba_size_d[ci]+=1

@cuda.jit
def find_change(nz_ba_size_d,nz_ba_size_cum_d,nz_ba_d,bound_data_ordered_d,scaled_shape,ba_change_d,ba_sign_d):
    ci=cuda.grid(1)
    if ci<len(nz_ba_size_d):
        n=nz_ba_size_cum_d[ci]
        s=nz_ba_size_d[ci]-2
        a=bound_data_ordered_d[nz_ba_d[n+s]-1]
        b=bound_data_ordered_d[nz_ba_d[n]-1]
        c=bound_data_ordered_d[nz_ba_d[n+1]-1]
        a0=int(a/scaled_shape[1])
        a1=a%scaled_shape[1]
        b0=int(b/scaled_shape[1])
        b1=b%scaled_shape[1]
        c0=int(c/scaled_shape[1])
        c1=c%scaled_shape[1]
            
        angle_pre=math.atan2(np.float64(a1-b1),np.float64(a0-b0))*180/math.pi
        angle_cur=math.atan2(np.float64(b1-c1),np.float64(b0-c0))*180/math.pi
        diff=angle_diff(angle_pre,angle_cur)
        ba_change_d[n]=diff
        if diff<0:
            ba_sign_d[n]=-1
        elif diff>0:
            ba_sign_d[n]=1
        n=nz_ba_size_cum_d[ci]+1
        s=0
        while 1:
            if s==nz_ba_size_d[ci]-2:
                break
            a=bound_data_ordered_d[nz_ba_d[n+s-1]-1]
            b=bound_data_ordered_d[nz_ba_d[n+s]-1]
            c=bound_data_ordered_d[nz_ba_d[n+s+1]-1]
            a0=int(a/scaled_shape[1])
            a1=a%scaled_shape[1]
            b0=int(b/scaled_shape[1])
            b1=b%scaled_shape[1]
            c0=int(c/scaled_shape[1])
            c1=c%scaled_shape[1]
            
            angle_pre=math.atan2(np.float64(a1-b1),np.float64(a0-b0))*180/math.pi
            angle_cur=math.atan2(np.float64(b1-c1),np.float64(b0-c0))*180/math.pi
            diff=angle_diff(angle_pre,angle_cur)
            ba_change_d[n+s]=diff
            if diff<0:
                ba_sign_d[n+s]=-1
            elif diff>0:
                ba_sign_d[n+s]=1
            s+=1

@cuda.jit(device=True)
def angle_diff(a,b):
    diff=b-a
    if diff>180:
        diff=diff-360
    elif diff<-180:
        diff=diff+360
    return diff


class Abstract:
    def __init__(self,bound_data_ordered_h,n_bounds,bound_abstract_h,shape_h,is_closed):
        # Inputs
        self.bound_data_ordered_h=bound_data_ordered_h
        self.n_bounds=n_bounds
        self.init_bound_abstract_h=bound_abstract_h
        self.bound_abstract_h=bound_abstract_h
        self.shape_h=shape_h
        self.is_closed=is_closed
        self.pd=np.full(1,np.inf,dtype=np.float32)
        self.pd_change=0
        if is_closed:
            self.ba_size_pre_hor=np.full(n_bounds,3,dtype=np.int32)
            self.pre_count=3
        else:
            self.ba_size_pre_hor=np.full(n_bounds,2,dtype=np.int32)
            self.pre_count=2
 
        self.nz_ba_pre_hor=get_non_zeros(bound_abstract_h)
        self.ba_sign_pre_h=[]
        
    def do_abstract_all(self,ba_threshold_pre):
        bound_data_ordered_d=cuda.to_device(self.bound_data_ordered_h)
        bound_abstract_pre_d=cuda.to_device(self.bound_abstract_h)
        shape_d=cuda.to_device(self.shape_h)
        nz_ba_pre_hor=get_non_zeros(self.bound_abstract_h)
        nz_ba_pre_hor_d=cuda.to_device(nz_ba_pre_hor)

            
        ba_size_pre_hor_d=cuda.to_device(self.ba_size_pre_hor)
        ba_size_cum_pre_hor_=np.cumsum(self.ba_size_pre_hor)
        ba_size_cum_pre_hor=np.delete(np.insert(ba_size_cum_pre_hor_,0,0),-1)
        ba_size_cum_pre_hor_d=cuda.to_device(ba_size_cum_pre_hor)
        
        ba_max_pd_pre=np.zeros([len(nz_ba_pre_hor),2],np.float64)
        ba_max_pd_pre_d=cuda.to_device(ba_max_pd_pre)
        ba_size_cum_pre_old_hor=ba_size_cum_pre_hor_[-1]
        #pre_count=self.pre_count
        while 1:
            find_ba_max_pd[math.ceil(len(nz_ba_pre_hor)/32),32](nz_ba_pre_hor_d,ba_size_pre_hor_d,bound_data_ordered_d,ba_max_pd_pre_d,shape_d)
            cuda.synchronize()
            ba_max_pd_pre_h=ba_max_pd_pre_d.copy_to_host()
            find_next_ba_all[math.ceil(len(self.ba_size_pre_hor)/32),32](ba_max_pd_pre_d,ba_size_pre_hor_d,ba_size_cum_pre_hor_d,bound_abstract_pre_d,ba_threshold_pre)
            cuda.synchronize()


            bound_abstract_pre_h=bound_abstract_pre_d.copy_to_host()
            nz_ba_pre_hor=get_non_zeros(bound_abstract_pre_h)
            nz_ba_pre_hor_d=cuda.to_device(nz_ba_pre_hor)
        
            ba_max_pd_pre=np.zeros([len(nz_ba_pre_hor),2],np.float64)
            ba_max_pd_pre_d=cuda.to_device(ba_max_pd_pre)

            ba_size_pre_hor=ba_size_pre_hor_d.copy_to_host()
            ba_size_cum_pre_hor_=np.cumsum(ba_size_pre_hor)
            ba_size_cum_pre_hor=np.delete(np.insert(ba_size_cum_pre_hor_,0,0),-1)
            ba_size_cum_pre_hor_d=cuda.to_device(ba_size_cum_pre_hor)
            
            if ba_size_cum_pre_hor_[-1]==ba_size_cum_pre_old_hor:
                ba_change_pre=np.zeros([len(nz_ba_pre_hor)],dtype=np.float64)
                ba_change_pre_d=cuda.to_device(ba_change_pre)
                ba_sign_pre=np.zeros([len(nz_ba_pre_hor)],dtype=np.int32)
                ba_sign_pre_d=cuda.to_device(ba_sign_pre)
                find_change[math.ceil(len(nz_ba_pre_hor)/32),32](ba_size_pre_hor_d,ba_size_cum_pre_hor_d,nz_ba_pre_hor_d,bound_data_ordered_d,shape_d,ba_change_pre_d,ba_sign_pre_d)
                cuda.synchronize()
                ba_change_pre_h=ba_change_pre_d.copy_to_host()
                ba_sign_pre_h=ba_sign_pre_d.copy_to_host()
                print("count=",self.pre_count,ba_size_cum_pre_hor_[-1])
                print("abstraction complete.")
                break
            else:
                ba_size_cum_pre_old_hor=ba_size_cum_pre_hor_[-1]
                self.pre_count+=1
        self.ba_size_pre_hor=ba_size_pre_hor
        self.nz_ba_pre_hor=nz_ba_pre_hor
        self.ba_sign_pre_h=ba_sign_pre_h
        self.bound_abstract_h=bound_abstract_pre_h
        
    def do_abstract_one(self,ba_threshold_pre):
        is_final=False
        bound_data_ordered_d=cuda.to_device(self.bound_data_ordered_h)
        bound_abstract_pre_d=cuda.to_device(self.bound_abstract_h)
        shape_d=cuda.to_device(self.shape_h)
        nz_ba_pre_hor=get_non_zeros(self.bound_abstract_h)
        nz_ba_pre_hor_d=cuda.to_device(nz_ba_pre_hor)
            
        ba_size_pre_hor_d=cuda.to_device(self.ba_size_pre_hor)
        ba_size_cum_pre_hor_=np.cumsum(self.ba_size_pre_hor)
        ba_size_cum_pre_hor=np.delete(np.insert(ba_size_cum_pre_hor_,0,0),-1)
        ba_size_cum_pre_hor_d=cuda.to_device(ba_size_cum_pre_hor)
        
        ba_max_pd_pre=np.zeros([len(nz_ba_pre_hor),2],np.float64)
        ba_max_pd_pre_d=cuda.to_device(ba_max_pd_pre)
        ba_size_cum_pre_old_hor=ba_size_cum_pre_hor_[-1]
        #pre_count=self.pre_count
        pd=np.zeros(1,dtype=np.float32)
        pd_d=cuda.to_device(pd)
        while 1:
            find_ba_max_pd[math.ceil(len(nz_ba_pre_hor)/32),32](nz_ba_pre_hor_d,ba_size_pre_hor_d,bound_data_ordered_d,ba_max_pd_pre_d,shape_d)
            cuda.synchronize()
            ba_max_pd_pre_h=ba_max_pd_pre_d.copy_to_host()
            find_next_ba[math.ceil(len(self.ba_size_pre_hor)/32),32](ba_max_pd_pre_d,ba_size_pre_hor_d,ba_size_cum_pre_hor_d,bound_abstract_pre_d,ba_threshold_pre,pd_d)
            cuda.synchronize()

            bound_abstract_pre_h=bound_abstract_pre_d.copy_to_host()
            nz_ba_pre_hor=get_non_zeros(bound_abstract_pre_h)
            nz_ba_pre_hor_d=cuda.to_device(nz_ba_pre_hor)
        
            ba_max_pd_pre=np.zeros([len(nz_ba_pre_hor),2],np.float64)
            ba_max_pd_pre_d=cuda.to_device(ba_max_pd_pre)

            ba_size_pre_hor=ba_size_pre_hor_d.copy_to_host()
            ba_size_cum_pre_hor_=np.cumsum(ba_size_pre_hor)
            ba_size_cum_pre_hor=np.delete(np.insert(ba_size_cum_pre_hor_,0,0),-1)
            ba_size_cum_pre_hor_d=cuda.to_device(ba_size_cum_pre_hor)
            
            if ba_size_cum_pre_hor_[-1]==ba_size_cum_pre_old_hor:
                is_final=True
                print("abstraction complete.")
            else:
                self.pre_count+=1
            ba_change_pre=np.zeros([len(nz_ba_pre_hor)],dtype=np.float64)
            ba_change_pre_d=cuda.to_device(ba_change_pre)
            ba_sign_pre=np.zeros([len(nz_ba_pre_hor)],dtype=np.int32)
            ba_sign_pre_d=cuda.to_device(ba_sign_pre)
            find_change[math.ceil(len(nz_ba_pre_hor)/32),32](ba_size_pre_hor_d,ba_size_cum_pre_hor_d,nz_ba_pre_hor_d,bound_data_ordered_d,shape_d,ba_change_pre_d,ba_sign_pre_d)
            cuda.synchronize()
            ba_change_pre_h=ba_change_pre_d.copy_to_host()
            ba_sign_pre_h=ba_sign_pre_d.copy_to_host()
            #print("count=",self.pre_count,ba_size_cum_pre_hor_[-1])
        
            ba_size_cum_pre_old_hor=ba_size_cum_pre_hor_[-1]
            break
        self.ba_size_pre_hor=ba_size_pre_hor
        self.nz_ba_pre_hor=nz_ba_pre_hor
        self.ba_sign_pre_h=ba_sign_pre_h
        self.bound_abstract_h=bound_abstract_pre_h
        return is_final
        
    def get_sign(self):
        return self.ba_sign_pre_h
    
    def reset_abstract(self):
        if self.is_closed:
            self.ba_size_pre_hor=np.full(self.n_bounds,3,dtype=np.int32)
            self.pre_count=3
        else:
            self.ba_size_pre_hor=np.full(self.n_bounds,2,dtype=np.int32)
            self.pre_count=2
        self.bound_abstract_h=self.init_bound_abstract_h
        self.nz_ba_pre_hor=get_non_zeros(self.init_bound_abstract_h)
        self.ba_sign_pre_h=[]
        
    def get_abstract(self):
        return self.nz_ba_pre_hor
    
    def get_pd_change(self):
        return self.pd_change
    def get_pd(self):
        return self.pd
    def get_abstract_size(self):
        return self.ba_size_pre_hor
