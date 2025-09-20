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

import numpy as np
from numba import cuda,int32
from collections import Counter
import pickle,os,math,itertools
from pyrebel.utils import *
from pyrebel.getnonzeros import *

@cuda.jit
def find_signatures_cuda(ba_sign_d,nz_ba_size_d,nz_ba_size_cum_d,ba_sign_array_d):
    ci=cuda.grid(1)
    if ci<len(nz_ba_size_d):
        #if nz_ba_size_d[ci]-1>layer_n:
        #    return
        for i in range(nz_ba_size_d[ci]-1):
            n=nz_ba_size_cum_d[ci]+i
            k1=int(4*(nz_ba_size_d[ci]-1)*(nz_ba_size_d[ci]-1-1)/2-4*3)
            #k2=int(4*(nz_ba_size_d[ci]-1)*(nz_ba_size_d[ci]-1+1)/2-4*3)
            n_first=n
            if n_first==nz_ba_size_cum_d[ci]:
                n_last=nz_ba_size_cum_d[ci]+nz_ba_size_d[ci]-2
            else:
                n_last=n_first-1
            #if ba_sign_d[n_last]==ba_sign_d[n_first]:
            #    continue
            s=1
            while 1:
                if ba_sign_d[n]>0:
                    ii1=1
                else:
                    ii1=0 
                if ii1==1:
                    ii1_inv=0
                else:
                    ii1_inv=1

                ba_sign_array_d[ci][k1+i][0]=1
                ba_sign_array_d[ci][k1+i][s-1+1]=pow(10,s-1+1)*ii1
                ba_sign_array_d[ci][k1+i][nz_ba_size_d[ci]]=pow(10,nz_ba_size_d[ci])
                
                ba_sign_array_d[ci][k1+(nz_ba_size_d[ci]-1)+i][0]=1
                ba_sign_array_d[ci][k1+(nz_ba_size_d[ci]-1)+i][s-1+1]=pow(10,s-1+1)*ii1_inv
                ba_sign_array_d[ci][k1+(nz_ba_size_d[ci]-1)+i][nz_ba_size_d[ci]]=pow(10,nz_ba_size_d[ci])
                
                ba_sign_array_d[ci][k1+(nz_ba_size_d[ci]-1)*2+i][0]=1
                ba_sign_array_d[ci][k1+(nz_ba_size_d[ci]-1)*2+i][nz_ba_size_d[ci]-s-1+1]=pow(10,nz_ba_size_d[ci]-s-1+1)*ii1
                ba_sign_array_d[ci][k1+(nz_ba_size_d[ci]-1)*2+i][nz_ba_size_d[ci]]=pow(10,nz_ba_size_d[ci])
                
                ba_sign_array_d[ci][k1+(nz_ba_size_d[ci]-1)*3+i][0]=1
                ba_sign_array_d[ci][k1+(nz_ba_size_d[ci]-1)*3+i][nz_ba_size_d[ci]-s-1+1]=pow(10,nz_ba_size_d[ci]-s-1+1)*ii1_inv
                ba_sign_array_d[ci][k1+(nz_ba_size_d[ci]-1)*3+i][nz_ba_size_d[ci]]=pow(10,nz_ba_size_d[ci])
                
                if s==nz_ba_size_d[ci]-1:
                    break
                if n==nz_ba_size_cum_d[ci]+nz_ba_size_d[ci]-2:
                    n=nz_ba_size_cum_d[ci]
                else:
                    n+=1
                s+=1
@cuda.jit
def find_signatures_cuda2(ba_sign_d,nz_ba_size_d,nz_ba_size_cum_d,ba_sign_array_d,cur_layer):
    ci=cuda.grid(1)
    if ci<len(nz_ba_size_d):
        #if nz_ba_size_d[ci]-1>layer_n:
        #    return
        n=nz_ba_size_cum_d[ci]
        s=1
        pos_count=0
        neg_count=0
        while 1:
            if ba_sign_d[n]>0:
                pos_count+=1
            else:
                neg_count+=1
            if s==nz_ba_size_d[ci]-1:
                break
            n+=1
            s+=1
        full_pos=pos_count==nz_ba_size_d[ci]-1
        full_neg=neg_count==nz_ba_size_d[ci]-1
        is_full=full_pos or full_neg
        for i in range(nz_ba_size_d[ci]-1):
            n=nz_ba_size_cum_d[ci]+i
            #k1=int(4*(nz_ba_size_d[ci]-1)*(nz_ba_size_d[ci]-1-1)/2-4*3)
            k1=int(4*(cur_layer)*(cur_layer-1)/2-4*3)
            #k2=int(4*(nz_ba_size_d[ci]-1)*(nz_ba_size_d[ci]-1+1)/2-4*3)
            n_first=n
            if n_first==nz_ba_size_cum_d[ci]:
                n_last=nz_ba_size_cum_d[ci]+nz_ba_size_d[ci]-2
            else:
                n_last=n_first-1
            if ba_sign_d[n_last]==ba_sign_d[n_first] and not is_full:
                continue
            s=1
            while 1:
                if ba_sign_d[n]>0:
                    ii1=1
                else:
                    ii1=0 
                if ii1==1:
                    ii1_inv=0
                else:
                    ii1_inv=1

                ba_sign_array_d[ci][k1+i][s-1]=ii1             
                ba_sign_array_d[ci][k1+cur_layer+i][s-1]=ii1_inv
                ba_sign_array_d[ci][k1+cur_layer*2+i][nz_ba_size_d[ci]-s-1]=ii1
                ba_sign_array_d[ci][k1+cur_layer*3+i][nz_ba_size_d[ci]-s-1]=ii1_inv
                
                if s==nz_ba_size_d[ci]-1:
                    break
                if n==nz_ba_size_cum_d[ci]+nz_ba_size_d[ci]-2:
                    n=nz_ba_size_cum_d[ci]
                else:
                    n+=1
                s+=1

class Learn:
    def __init__(self,layer_n,n_blobs,next_layer):
        self.cur_sign_list_dict={}
        self.n_blobs=n_blobs
        self.layer_n=layer_n
        self.next_layer=next_layer
        if not os.path.exists('know_base.pkl'):
            fp=open('know_base.pkl','x')
            fp.close()
        with open("know_base.pkl","rb") as fpr:
            try:
                self.know_base=pickle.load(fpr)
            except EOFError:
                self.know_base={}
        self.ba_sign_array2_h=np.full([n_blobs,int(((layer_n*(layer_n+1)/2)-3)*4),layer_n],2,dtype=np.int8)
        self.ba_sign_array_h=np.zeros([n_blobs,int(((layer_n*(layer_n+1)/2)-3)*4),layer_n+2],dtype=np.int64)
        self.nz_ba_sign_array2_str_signs=[]
        self.count_cum=[]
        self.count_h=[]

    def init_signatures(self):
        ba_sign_array2_d=cuda.to_device(self.ba_sign_array2_h)
        mask=run_cuda_duplicate_detection_large(self.ba_sign_array2_h)
        count=np.zeros(len(self.ba_sign_array2_h),dtype=np.int32)
        count_d=cuda.to_device(count)
        mask_d=cuda.to_device(mask)
        threads_per_block_2d=(32,32)
        blocks_per_grid_x=math.ceil(self.ba_sign_array2_h.shape[0]/threads_per_block_2d[0])
        blocks_per_grid_y=math.ceil(self.ba_sign_array2_h.shape[1]/threads_per_block_2d[1])
        get_row_wise_count[(blocks_per_grid_x,blocks_per_grid_y),threads_per_block_2d](mask_d,count_d)
        cuda.synchronize()
        
        fill_axis1[(blocks_per_grid_x,blocks_per_grid_y),threads_per_block_2d](ba_sign_array2_d,mask_d,3)
        cuda.synchronize()
        
        ba_sign_array2_flat=ba_sign_array2_d.copy_to_host().flatten().astype('int32')
        nz_ba_sign_array2_flat=get_non_zeros(ba_sign_array2_flat,3)
        n_signs=len(nz_ba_sign_array2_flat)/self.layer_n
        #print(n_signs)
        self.count_h=count_d.copy_to_host()
        count_cum_=np.cumsum(self.count_h)
        self.count_cum=np.delete(np.insert(count_cum_,0,0),-1)
        
        nz_ba_sign_array2_str=[''.join(map(str, nz_ba_sign_array2_flat[i:i+self.layer_n])) for i in range(0, len(nz_ba_sign_array2_flat), self.layer_n)]
        self.nz_ba_sign_array2_str_signs = [s.split('2')[0] for s in nz_ba_sign_array2_str]
        #print(count_cum)
        #print(count_h)
    
    def find_signatures2(self,ba_sign_h,nz_ba_size_h):
        if self.next_layer>self.layer_n:
            return 1
        nz_ba_size_cum_=np.cumsum(nz_ba_size_h)
        nz_ba_size_cum=np.delete(np.insert(nz_ba_size_cum_,0,0),-1)
        nz_ba_size_cum_d=cuda.to_device(nz_ba_size_cum)
        nz_ba_size_d=cuda.to_device(nz_ba_size_h)
        ba_sign_d=cuda.to_device(ba_sign_h)
        ba_sign_array2_d=cuda.to_device(self.ba_sign_array2_h)
        find_signatures_cuda2[len(nz_ba_size_h),1](ba_sign_d,nz_ba_size_d,nz_ba_size_cum_d,ba_sign_array2_d,self.next_layer)
        cuda.synchronize()
        self.next_layer+=1
        self.ba_sign_array2_h=ba_sign_array2_d.copy_to_host()
        return 0
    
    def find_signatures(self,ba_sign_h,nz_ba_size_h):
        nz_ba_size_cum_=np.cumsum(nz_ba_size_h)
        nz_ba_size_cum=np.delete(np.insert(nz_ba_size_cum_,0,0),-1)
        nz_ba_size_cum_d=cuda.to_device(nz_ba_size_cum)
        nz_ba_size_d=cuda.to_device(nz_ba_size_h)
        ba_sign_d=cuda.to_device(ba_sign_h)
        ba_sign_array_d=cuda.to_device(self.ba_sign_array_h)
        find_signatures_cuda[len(nz_ba_size_h),1](ba_sign_d,nz_ba_size_d,nz_ba_size_cum_d,ba_sign_array_d)
        cuda.synchronize()
        self.ba_sign_array_h=ba_sign_array_d.copy_to_host()           
    
    def learn_sym(self,blobs_i,sign_name,channel):        
        n=0
        learned_signs=list()

        if len(blobs_i)==0:
            return [] 
        elif len(blobs_i)==1:
            print("Single blob.")
            blobs_indices=blobs_i
            for blob_i in range(len(blobs_indices)):
                for sign_i in range(self.count_cum[blobs_indices[blob_i]],self.count_cum[blobs_indices[blob_i]]+self.count_h[blobs_indices[blob_i]]):
                    cur_sign=self.nz_ba_sign_array2_str_signs[sign_i]
                    if cur_sign in self.know_base:
                        if channel in self.know_base[cur_sign]:
                            if sign_name in self.know_base[cur_sign][channel]:
                                self.know_base[cur_sign][channel][sign_name]+=1
                            else:
                                self.know_base[cur_sign][channel][sign_name]=1
                        else:
                            self.know_base[cur_sign][channel]={sign_name:1}
                            learned_signs.append(cur_sign)
                            n+=1 
                    else:
                        self.know_base[cur_sign]={channel:{sign_name:1}}
                        #print(cur_sign)
                        learned_signs.append(cur_sign) 
                        n+=1
                        
        else:
            print("Multiple blobs.")
            blobs_indices=blobs_i
            top_n=1
            recognized=list()
            blob_i_counter=list()
            for blob_i in range(len(blobs_indices)):
                recognized=list()
                blob_i_counter.append(list())
                for sign_i in range(self.count_cum[blobs_indices[blob_i]],self.count_cum[blobs_indices[blob_i]]+self.count_h[blobs_indices[blob_i]]):
                    cur_sign=self.nz_ba_sign_array2_str_signs[sign_i]
                    if cur_sign in self.know_base:
                        if channel in self.know_base[cur_sign]:
                            symbol_recognized=self.know_base[cur_sign][channel].keys()
                            recognized+=symbol_recognized
                blob_i_counter[blob_i]=Counter(recognized).most_common(top_n)
            sym=list(itertools.chain(*blob_i_counter))
            #sym.sort(key=lambda x: x[1],reverse=True)
            #print(sym)
            for s in sym:
                if s[0] in self.know_base:
                    if channel in self.know_base[s[0]]:
                        if sign_name not in self.know_base[s[0]][channel]:
                            self.know_base[s[0]][channel][sign_name]=s[1]
                            learned_signs.append(s)  
                    else:
                        self.know_base[s[0]][channel]={sign_name:s[1]}
                        learned_signs.append(s)                 
                else:
                    self.know_base[s[0]]={channel: {sign_name:s[1]}}
                    learned_signs.append(s)
            #learned_signs.sort(key=lambda x: x[1],reverse=True)
            learned_signs=blob_i_counter
        return learned_signs    
     
    def learn_one(self,blob_i,sign_name):
        n=0
        learned_signs=list()
        #print(self.ba_sign_array2_h[blob_i].shape)
        ba_sign_array2_blob_h=self.ba_sign_array2_h[blob_i].reshape((1,)+self.ba_sign_array2_h[blob_i].shape)
        ba_sign_array2_blob_d=cuda.to_device(ba_sign_array2_blob_h)
        mask=run_cuda_duplicate_detection_large(ba_sign_array2_blob_h)
        
        mask_d=cuda.to_device(mask)
        threads_per_block_2d=(32,32)
        blocks_per_grid_x=math.ceil(ba_sign_array2_blob_h.shape[0]/threads_per_block_2d[0])
        blocks_per_grid_y=math.ceil(ba_sign_array2_blob_h.shape[1]/threads_per_block_2d[1])
        fill_axis1[(blocks_per_grid_x,blocks_per_grid_y),threads_per_block_2d](ba_sign_array2_blob_d,mask_d,3)
        cuda.synchronize()
        
        ba_sign_array2_blob_h=ba_sign_array2_blob_d.copy_to_host()
        ba_sign_array2_blob_flat=ba_sign_array2_blob_h.flatten()
        nz_ba_sign_array2_blob_flat=get_non_zeros(ba_sign_array2_blob_flat.astype('int32'),3)
        ba_sign_array2_blob_h=nz_ba_sign_array2_blob_flat.reshape([int(len(nz_ba_sign_array2_blob_flat)/self.layer_n),self.layer_n])
        #print(ba_sign_array2_blob_h.shape)
        
        ba_sign_array2_blob_h=ba_sign_array2_blob_h.astype(str)
        join=lambda x: np.asarray(''.join(x).split('2')[0],dtype=object)
        ba_sign_array2_blob_h=np.apply_along_axis(join,1,ba_sign_array2_blob_h)
        ba_sign_array2_blob_h=np.unique(ba_sign_array2_blob_h)
        for cur_sign in ba_sign_array2_blob_h:
            if cur_sign in self.know_base:
                if sign_name in self.know_base[cur_sign]:
                    self.know_base[cur_sign][sign_name]+=1
                else:
                    self.know_base[cur_sign][sign_name]=1
                    learned_signs.append(cur_sign)
                    n+=1 
            else:
                self.know_base[cur_sign]={sign_name:1}
                #print(cur_sign)
                learned_signs.append(cur_sign) 
                n+=1
        return learned_signs
           
    def learn2(self,blob_i,sign_name):
        n=0
        #sign_sum=self.ba_sign_array_h.sum(axis=2)
        #sign_sum_unique=np.unique(sign_sum,axis=1)
        learned_signs=list()
        self.ba_sign_array2_h=self.ba_sign_array2_h.astype(str)
        join=lambda x: np.asarray(''.join(x).split('2')[0],dtype=object)
        self.ba_sign_array2_h=np.apply_along_axis(join,2,self.ba_sign_array2_h)
        unique=lambda x: np.asarray(set(x))
        self.ba_sign_array2_h=np.apply_along_axis(unique,1,self.ba_sign_array2_h)
        
        for cur_sign in self.ba_sign_array2_h[blob_i]:
            if cur_sign in self.know_base:
                if sign_name in self.know_base[cur_sign]:
                    self.know_base[cur_sign][sign_name]+=1
                else:
                    self.know_base[cur_sign][sign_name]=1
                    learned_signs.append(cur_sign)
                    n+=1 
            else:
                self.know_base[cur_sign]={sign_name:1}
                #print(cur_sign)
                learned_signs.append(cur_sign) 
                n+=1
        return learned_signs
    
    def learn(self,blob_i,sign_name):
        n=0
        sign_sum=self.ba_sign_array_h.sum(axis=2)
        sign_sum_unique=np.unique(sign_sum,axis=1)
        learned_signs=list()
        
        for cur_sign in sign_sum_unique[blob_i]:
            if str(cur_sign) in self.know_base:
                if sign_name in self.know_base[str(cur_sign)]:
                    self.know_base[str(cur_sign)][sign_name]+=1
                else:
                    self.know_base[str(cur_sign)][sign_name]=1
                    #print(cur_sign)
                    learned_signs.append(str(cur_sign))
                    n+=1
            else:
                self.know_base[str(cur_sign)]={sign_name:1}
                #print(cur_sign)
                learned_signs.append(str(cur_sign)) 
                n+=1
        return learned_signs
    
    def get_sign_array(self):
        return self.ba_sign_array_h
    def get_sign_array2(self):
        return self.ba_sign_array2_h

    def recognize_sym(self,blobs_i,top_n,channel):
        recognized=list()
        blob_i_counter=list()
        if len(blobs_i)==0:
            blobs_indices=range(len(self.count_h))
            
        elif len(blobs_i)==1:
            blobs_indices=blobs_i
            for blob_i in range(len(blobs_indices)):
                recognized=list()
                blob_i_counter.append(list())
                for sign_i in range(self.count_cum[blobs_indices[blob_i]],self.count_cum[blobs_indices[blob_i]]+self.count_h[blobs_indices[blob_i]]):
                    cur_sign=self.nz_ba_sign_array2_str_signs[sign_i]
                    if cur_sign in self.know_base:
                        if channel in self.know_base[cur_sign]:
                            symbol_recognized=self.know_base[cur_sign][channel].keys()
                            recognized+=symbol_recognized
                blob_i_counter[blob_i]=Counter(recognized).most_common(top_n)
            sym2=blob_i_counter[0]
            return [blob_i_counter,[sym2]]
        else:
            blobs_indices=blobs_i
            for blob_i in range(len(blobs_indices)):
                recognized=list()
                blob_i_counter.append(list())
                for sign_i in range(self.count_cum[blobs_indices[blob_i]],self.count_cum[blobs_indices[blob_i]]+self.count_h[blobs_indices[blob_i]]):
                    cur_sign=self.nz_ba_sign_array2_str_signs[sign_i]
                    if cur_sign in self.know_base:
                        if channel in self.know_base[cur_sign]:
                            symbol_recognized=self.know_base[cur_sign][channel].keys()
                            recognized+=symbol_recognized
                blob_i_counter[blob_i]=Counter(recognized).most_common(top_n)
            sym=list(itertools.chain(*blob_i_counter))
            
            #sym.sort(key=lambda x: x[1],reverse=True)
            #print(sym)
            recognized2=list()
            multiple=0
            for s in sym:
                if len(s[0])>1:
                    multiple+=1
                if s[0] in self.know_base:
                    sym_recognized=list(self.know_base[s[0]][channel].keys())*s[1]
                    recognized2+=sym_recognized
                    
            if channel=="image" and multiple==0:
                recognized2.append("".join(i[0] for i in sym))
            sym2=Counter(recognized2).most_common(top_n)        
            return [blob_i_counter,[sym2]]

    def recognize_one(self,blob_i,top_n):
        recognized=list()
        if len(self.ba_sign_array_h)>blob_i:
            #print(self.ba_sign_array2_h[blob_i].shape)
            ba_sign_array2_blob_h=self.ba_sign_array2_h[blob_i].reshape((1,)+self.ba_sign_array2_h[blob_i].shape)
            ba_sign_array2_blob_d=cuda.to_device(ba_sign_array2_blob_h)
            mask=run_cuda_duplicate_detection_large(ba_sign_array2_blob_h)

            mask_d=cuda.to_device(mask)
            threads_per_block_2d=(32,32)
            blocks_per_grid_x=math.ceil(ba_sign_array2_blob_h.shape[0]/threads_per_block_2d[0])
            blocks_per_grid_y=math.ceil(ba_sign_array2_blob_h.shape[1]/threads_per_block_2d[1])
            fill_axis1[(blocks_per_grid_x,blocks_per_grid_y),threads_per_block_2d](ba_sign_array2_blob_d,mask_d,3)
            cuda.synchronize()
            
            ba_sign_array2_blob_h=ba_sign_array2_blob_d.copy_to_host()
            ba_sign_array2_blob_flat=ba_sign_array2_blob_h.flatten()
            nz_ba_sign_array2_blob_flat=get_non_zeros(ba_sign_array2_blob_flat.astype('int32'),3)
            ba_sign_array2_blob_h=nz_ba_sign_array2_blob_flat.reshape([int(len(nz_ba_sign_array2_blob_flat)/self.layer_n),self.layer_n])
            
            #print(ba_sign_array2_blob_h.shape)
            ba_sign_array2_blob_h=ba_sign_array2_blob_h.astype(str)
            join=lambda x: np.asarray(''.join(x).split('2')[0],dtype=object)
            ba_sign_array2_blob_h=np.apply_along_axis(join,1,ba_sign_array2_blob_h)
            ba_sign_array2_blob_h=np.unique(ba_sign_array2_blob_h)
            for cur_sign in ba_sign_array2_blob_h:
                if cur_sign in self.know_base:
                    symbol_recognized=self.know_base[cur_sign].keys()
                    recognized+=symbol_recognized
            blob_i_counter=Counter(recognized)
            return dict(blob_i_counter.most_common(top_n))
        return {}        
        
    def recognize2(self,blob_i,top_n):
        recognized=list()
        if len(self.ba_sign_array_h)>blob_i:
            #sign_sum=self.ba_sign_array_h.sum(axis=2)
            #sign_sum_blob=self.ba_sign_array_h[blob_i].sum(axis=1)
            #sign_sum_unique=np.unique(sign_sum,axis=1)
            self.ba_sign_array2_h=self.ba_sign_array2_h.astype(str)
            join=lambda x: np.asarray(''.join(x).split('2')[0],dtype=object)
            self.ba_sign_array2_h=np.apply_along_axis(join,2,self.ba_sign_array2_h)
            unique=lambda x: np.asarray(set(x))
            self.ba_sign_array2_h=np.apply_along_axis(unique,1,self.ba_sign_array2_h)
            for cur_sign in self.ba_sign_array2_h[blob_i]:
                if cur_sign in self.know_base:
                    symbol_recognized=self.know_base[cur_sign].keys()
                    recognized+=symbol_recognized
            blob_i_counter=Counter(recognized)
            return dict(blob_i_counter.most_common(top_n))
        return {}
            
    def recognize(self,blob_i,top_n):
        recognized=list()
        if len(self.ba_sign_array_h)>blob_i:
            sign_sum=self.ba_sign_array_h.sum(axis=2)
            #sign_sum_blob=self.ba_sign_array_h[blob_i].sum(axis=1)
            #sign_sum_unique=np.unique(sign_sum,axis=1)
            for cur_sign in sign_sum[blob_i]:
                if str(cur_sign) in self.know_base:
                    symbol_recognized=self.know_base[str(cur_sign)].keys()
                    recognized+=symbol_recognized
            blob_i_counter=Counter(recognized)
            return dict(blob_i_counter.most_common(top_n))
        return {}
        
    def write_know_base(self):
        with open('know_base.pkl','wb') as fpw:
            pickle.dump(self.know_base,fpw)
        
    def get_know_base(self):
        return self.know_base
