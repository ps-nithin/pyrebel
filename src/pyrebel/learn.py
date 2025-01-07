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
from collections import Counter
import pickle,os

def right_rotate(lst, n, inv_sign):
    # Convert the list to a numpy array
    arr = np.array(lst)
    arr=arr*inv_sign
    # Use np.roll to shift the elements to the right
    arr = np.roll(arr, n)
    # Convert the numpy array back to a list
    arr = arr.tolist()
    # Return the rotated list
    return arr

def inverse_sign(string):
    inv=list()
    for i in string:
        if int(i)==0:
            inv.append("1")
        else:
            inv.append("0")
    return ''.join(inv) 
    
class Learn:
    def __init__(self):
        self.cur_sign_list_dict={}
        if not os.path.exists('know_base.pkl'):
            fp=open('know_base.pkl','x')
            fp.close()
        with open("know_base.pkl","rb") as fpr:
            try:
                self.know_base=pickle.load(fpr)
            except EOFError:
                self.know_base={}
        
    def get_signatures(self,ba_sign_h,nz_ba_size_h):
        nz_ba_size_cum_=np.cumsum(nz_ba_size_h)
        nz_ba_size_cum=np.delete(np.insert(nz_ba_size_cum_,0,0),-1)
        if len(self.cur_sign_list_dict)==0:
            for i in range(len(nz_ba_size_h)):
                self.cur_sign_list_dict[i]=set()
        inv_sign=1
        for blob_i in range(len(nz_ba_size_h)):
            select_ba_sign=ba_sign_h[nz_ba_size_cum[blob_i]:nz_ba_size_cum[blob_i]+nz_ba_size_h[blob_i]-1]
            #print("blob:",i,end=" ")
            #print("layer:",n,end=" ")
            if len(select_ba_sign)==3:
                if select_ba_sign[0]<0:
                    inv_sign=-1
            for i in range(len(select_ba_sign)):
                cur_sign=right_rotate(select_ba_sign,i,inv_sign)
                sign=''.join("0" if sign_<0 else "1" for sign_ in cur_sign)
                if sign[0]=="1" and sign[0]!=sign[-1] or len(sign)==3:
                    #print(sign,end=" ")
                    self.cur_sign_list_dict[blob_i].add(sign)
                    self.cur_sign_list_dict[blob_i].add(sign[::-1])
                    sign_inverse=inverse_sign(sign)
                    self.cur_sign_list_dict[blob_i].add(sign_inverse)
                    self.cur_sign_list_dict[blob_i].add(sign_inverse[::-1])
                    
    def learn(self,blob_i,sign_name):
        n=0
        for cur_sign in self.cur_sign_list_dict[blob_i]:
            if cur_sign in self.know_base:
                if sign_name in self.know_base[cur_sign]:
                    self.know_base[cur_sign][sign_name]+=1
                else:
                    self.know_base[cur_sign][sign_name]=1
                    #print(cur_sign)
                    n+=1
            else:
                self.know_base[cur_sign]={sign_name:1}
                #print(cur_sign) 
                n+=1
        return n                              
                        
    def recognize_symbols(self,blob_i,top_n):
        recognized=list()
        for cur_sign in self.cur_sign_list_dict[blob_i]:
            if cur_sign in self.know_base:
                symbol_recognized=self.know_base[cur_sign].keys()
                recognized+=symbol_recognized
        blob_i_counter=Counter(recognized)
        return dict(blob_i_counter.most_common(top_n))
        
    def write_know_base(self):
        with open('know_base.pkl','wb') as fpw:
            pickle.dump(self.know_base,fpw)
        
    def get_know_base(self):
        return self.know_base
