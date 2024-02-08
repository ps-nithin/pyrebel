#!/usr/bin/env python3
#
# Copyright (c) 2024, Nithin PS. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
from __future__ import print_function
import sys
import math
from math import sqrt
import argparse
import time
from jetson_utils import videoSource, videoOutput, Log
from jetson_utils import cudaAllocMapped,cudaConvertColor
from jetson_utils import cudaToNumpy,cudaDeviceSynchronize,cudaFromNumpy
from numba import cuda
import numpy as np
import cmath
from PIL import Image

# create video sources & outputs
input = videoSource("csi://0", options={'width':320,'height':240,'framerate':30,'flipMethod':'rotate-270'})
output = videoOutput("", argv=sys.argv)

out_loc="output.png"

# process frames until EOS or the user exits
def main():
    n=0
    while True:
        start_time=time.time()
        init_time=time.time()
        if len(sys.argv)==1:
            # capture the next image
            img = input.Capture()
            if img is None: # timeout
                continue  
            img_gray=convert_color(img,'gray8')
            img_array=cudaToNumpy(img_gray)
            cudaDeviceSynchronize()
            img_array=img_array.reshape(1,img_array.shape[0],img_array.shape[1])[0].astype('int')
        else:
            img_array=open_image(sys.argv[1]).astype('int')

        
        """
        img_array=np.full([5,5],0,dtype=np.int)
        img_array[1][2]=5
        img_array[1][3]=5
        img_array[2][2]=5
        img_array[2][3]=5
        img_array[2][1]=5
        """
        img_array_d=cuda.to_device(img_array)
        #print(img_array)
        threadsperblock=(16,16)
        blockspergrid_x=math.ceil(img_array.shape[0]/threadsperblock[0])
        blockspergrid_y=math.ceil(img_array.shape[1]/threadsperblock[1])
        blockspergrid=(blockspergrid_x,blockspergrid_y)
        img_fenced_d=img_array_d
        fence_image[blockspergrid,threadsperblock](img_array_d,img_fenced_d)

        scaled_shape=np.array([img_array.shape[0]*3,img_array.shape[1]*3])
        scaled_shape_d=cuda.to_device(scaled_shape)
        img_scaled_d=cuda.device_array(scaled_shape,dtype=np.int)
        scale_img_cuda[blockspergrid,threadsperblock](img_fenced_d,img_scaled_d)
        cuda.synchronize()
        img_scaled_h=img_scaled_d.copy_to_host()
        
        threadsperblock=(16,16)
        blockspergrid_x=math.ceil(scaled_shape[0]/threadsperblock[0])
        blockspergrid_y=math.ceil(scaled_shape[1]/threadsperblock[1])
        blockspergrid=(blockspergrid_x,blockspergrid_y)
        img_boundary=np.full(scaled_shape,500,dtype=np.int)
        img_boundary_d=cuda.to_device(img_boundary)
        read_bound_cuda[blockspergrid,threadsperblock](img_scaled_d,img_boundary_d)
        cuda.synchronize()
        bound_info=np.zeros([scaled_shape[0]*scaled_shape[1],2],dtype=np.int)
        bound_info_d=cuda.to_device(bound_info)
        threadsperblock=(16,16)
        blockspergrid_x=math.ceil(scaled_shape[0]/threadsperblock[0])
        blockspergrid_y=math.ceil(scaled_shape[1]/threadsperblock[1])
        blockspergrid=(blockspergrid_x,blockspergrid_y)
        get_bound_cuda2[blockspergrid,threadsperblock](img_boundary_d,bound_info_d)
        cuda.synchronize()
        
        img_boundary_h=img_boundary_d.copy_to_host()
        binfo=bound_info_d.copy_to_host()
        a=binfo.transpose()[0]
        s=binfo.transpose()[1]
        nz_a=get_non_zeros(a)
        nz_s=get_non_zeros(s)
        #nz=np.column_stack((nz_a,nz_s))
        #nz_sort=nz[nz[:,1].argsort()]
        nz_s_cum_=np.cumsum(nz_s)
        nz_s_cum=np.delete(np.insert(nz_s_cum_,0,0),-1)
        nz_s_cum_d=cuda.to_device(nz_s_cum)
        nz_a_d=cuda.to_device(nz_a)
        nz_s_d=cuda.to_device(nz_s)

        nz_si_d=cuda.to_device(nz_s)
        increment_by_one[len(nz_s),1](nz_si_d)
        nz_si=nz_si_d.copy_to_host()
        nz_si_cum_=np.cumsum(nz_si)
        nz_si_cum=np.delete(np.insert(nz_si_cum_,0,0),-1)
        nz_si_cum_d=cuda.to_device(nz_si_cum)

        print("len(bound_data_d)=",nz_s_cum_[-1])
        print("len(bound_data_order_d)=",nz_si_cum_[-1])
        bound_data_d=cuda.device_array([nz_s_cum_[-1]],dtype=np.int)
        get_bound_data_init[math.ceil(len(nz_a)/256),256](nz_a_d,nz_s_cum_d,img_boundary_d,bound_data_d)
        cuda.synchronize()

        dist_data_d=cuda.device_array([nz_s_cum_[-1]],dtype=np.float)
        get_dist_data_init[math.ceil(nz_s_cum_[-1]/256),256](bound_data_d,img_boundary_d,dist_data_d)
        cuda.synchronize()
        
        max_dist_d=cuda.device_array([len(nz_s),2],dtype=np.int)
        get_max_dist[math.ceil(len(nz_s)/1),1](nz_s_cum_d,nz_s_d,bound_data_d,dist_data_d,max_dist_d)
        cuda.synchronize()

        bound_data_ordered_d=cuda.device_array([nz_si_cum_[-1]],dtype=np.int)
        bound_abstract=np.zeros([nz_si_cum_[-1]],dtype=np.int)
        bound_abstract_d=cuda.to_device(bound_abstract)
        bound_threshold=np.zeros([nz_si_cum_[-1]],dtype=np.float64)
        bound_threshold_d=cuda.to_device(bound_threshold)
        get_bound_data_order[math.ceil(len(nz_a)/256),256](max_dist_d,nz_si_cum_d,img_boundary_d,bound_abstract_d,bound_data_ordered_d,bound_threshold_d)
        cuda.synchronize()
        bound_threshold_h=bound_threshold_d.copy_to_host()
        bound_abstract_h=bound_abstract_d.copy_to_host()
        nz_ba_h=get_non_zeros(bound_abstract_h)
        nz_ba_d=cuda.to_device(nz_ba_h)
        nz_ba_pre_size=len(nz_ba_h)
        max_pd_h=np.zeros([1],np.float64)
        max_pd_d=cuda.to_device(max_pd_h)
        while 1:
            find_detail[len(nz_ba_h),1](nz_ba_d,bound_threshold_d,bound_abstract_d,bound_data_ordered_d,max_pd_d,scaled_shape_d)
            cuda.synchronize()
            bound_abstract_h=bound_abstract_d.copy_to_host()
            nz_ba_h=get_non_zeros(bound_abstract_h)
            nz_ba_d=cuda.to_device(nz_ba_h)
            if len(nz_ba_h)==nz_ba_pre_size:
                break
            nz_ba_pre_size=len(nz_ba_h)

        out_image=np.zeros(scaled_shape,dtype=np.int)
        out_image_d=cuda.to_device(out_image)
        
        
        draw_pixels_cuda(bound_data_ordered_d,100,out_image_d)
        decrement_by_one[len(nz_ba_h),1](nz_ba_d)
        draw_pixels_from_indices_cuda(nz_ba_d,bound_data_ordered_d,255,out_image_d)
        
        out_image_h=out_image_d.copy_to_host()
        #img_boundary_h=img_boundary_d.copy_to_host()
        #print(img_boundary_h)
        img_boundary_cuda=cudaFromNumpy(out_image_h)
        img_boundary_cuda_rgb=convert_color(img_boundary_cuda,'rgb8')
        output_png=cudaToNumpy(img_boundary_cuda_rgb)
        write_image(out_loc,output_png)
        # render the image
        output.Render(img_boundary_cuda_rgb)
        # exit on input/output EOS
        #if not input.IsStreaming() or not output.IsStreaming():
        #    break
        n+=1
        print("Finished in total of",time.time()-init_time,"seconds at",float(1/(time.time()-init_time)),"fps count=",n)

def main2():
    a=np.zeros(3000000,dtype=np.int64)
    for i in range(3200,65000,2000):
        a[i]=i
    nz=get_non_zeros(a)
    print(nz)

@cuda.jit
def find_detail(nz_ba_d,bound_threshold_d,bound_abstract_d,bound_data_ordered_d,max_pd_d,scaled_shape):
    ci=cuda.grid(1)
    if ci<len(nz_ba_d)-1:
        a=bound_data_ordered_d[nz_ba_d[ci]-1]
        b=bound_data_ordered_d[nz_ba_d[ci+1]-1]
        a0=int(a/scaled_shape[1])
        a1=a%scaled_shape[1]
        b0=int(b/scaled_shape[1])
        b1=b%scaled_shape[1]
        threshold=bound_threshold_d[nz_ba_d[ci]]
        #threshold=cmath.sqrt(float(pow(b0-a0,2)+pow(b1-a1,2))).real/8
        n=nz_ba_d[ci]+1
        pd_max=0.0
        pd_max_i=n
        while 1:
            if n==nz_ba_d[ci+1]:
                break
            c=bound_data_ordered_d[n-1]
            c0=int(c/scaled_shape[1])
            c1=c%scaled_shape[1]
            pd=abs((a1-b1)*(a0-c0)-(a0-b0)*(a1-c1))/cmath.sqrt(pow(a1-b1,2)+pow(a0-b0,2)).real

            if pd>pd_max:
                pd_max=pd
                pd_max_i=n
            n+=1
        cuda.atomic.max(max_pd_d,0,pd_max)
        cuda.syncthreads()
        if max_pd_d[0]==pd_max and pd_max>threshold:
            bound_abstract_d[pd_max_i]=pd_max_i
        max_pd_d[0]=0.0




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


@cuda.jit
def get_first_pixel(nz_s_cum_d,bound_data_d,first_pixel_d):
    ci=cuda.grid(1)
    if ci<len(bound_data_d):
        n=nz_s_cum_d[ci]
        first_pixel_d[ci]=bound_data_d[n]


def get_bound_from_seed(index,tmp_img):
    bound=list()
    y,x=i_to_p(index,tmp_img.shape)
    r=y
    c=x
    color=tmp_img[r][c]
    n=0
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
        n+=1
        bound.append(p_to_i([r,c],tmp_img.shape))
        if r==y and c==x:
            break
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
    return bound

@cuda.jit
def get_max_dist(nz_s_cum_d,nz_s_d,bound_data_d,dist_data_d,max_dist_d):
    ci=cuda.grid(1)
    if ci<len(nz_s_d):
        n=nz_s_cum_d[ci]
        s=0
        d_max=dist_data_d[n]
        d_max_i=n
        while 1:
            s+=1
            if dist_data_d[n]>d_max:
                d_max=dist_data_d[n]
                d_max_i=n
            if s==nz_s_d[ci]:
                break
            n+=1
        n=nz_s_cum_d[ci]
        s=0
        while 1:
            s+=1
            if dist_data_d[n]==d_max and n!=d_max_i:
                d_max2=dist_data_d[n]
                d_max_i2=n
            if s==nz_s_d[ci]:
                break
            n+=1

        max_dist_d[ci][0]=bound_data_d[d_max_i]
        max_dist_d[ci][1]=bound_data_d[d_max_i2]

@cuda.jit
def get_dist_data_init(bound_data_d,tmp_img,dist_data_d):
    ci=cuda.grid(1)
    if ci<len(bound_data_d):
        index=bound_data_d[ci]
        y=int(index/tmp_img.shape[1])
        x=index%tmp_img.shape[1]
        r=y
        c=x
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
        dist_data_d[ci]=d_max

@cuda.jit
def get_bound_data_order(nz_a_max_dist,nz_s,tmp_img,init_bound_abstract,bound_data_order_d,bound_threshold_d):
    ci=cuda.grid(1)
    if ci<len(nz_a_max_dist):
        index=nz_a_max_dist[ci][0]
        index2=nz_a_max_dist[ci][1]
        y=int(index/tmp_img.shape[1])
        x=index%tmp_img.shape[1]
        y2=int(index2/tmp_img.shape[1])
        x2=index2%tmp_img.shape[1]
        threshold_ratio=16
        threshold=sqrt(float(pow(y2-y,2)+pow(x2-x,2)))/threshold_ratio
        r=y
        c=x
        color=tmp_img[r][c]
        n=nz_s[ci]
        init_bound_abstract[n]=n+1
        bound_threshold_d[n]=threshold
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
                break
            n+=1
            bound_data_order_d[n]=r*tmp_img.shape[1]+c
            bound_threshold_d[n]=threshold

            if y2==r and x2==c:
                init_bound_abstract[n]=n+1
                
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

from numba import int32
BSP2=9
BLOCK_SIZE=2**BSP2
@cuda.jit('void(int32[:], int32[:], int32[:], int32, int32)')
def prefix_sum_nzmask_block(a,b,s,nzm,length):
    ab=cuda.shared.array(shape=(BLOCK_SIZE),dtype=int32)
    tid=cuda.blockIdx.x*cuda.blockDim.x+cuda.threadIdx.x
    if tid<length:
        if nzm==1:
            ab[cuda.threadIdx.x]=int32(a[tid]!=0)
        else:
            ab[cuda.threadIdx.x]=int32(a[tid])
    for j in range(0,BSP2):
        i=2**j
        cuda.syncthreads()
        if i<=cuda.threadIdx.x:
            temp=ab[cuda.threadIdx.x]
            temp+=ab[cuda.threadIdx.x-i]
        cuda.syncthreads()
        if i<=cuda.threadIdx.x:
            ab[cuda.threadIdx.x]=temp
    if tid<length:
        b[tid]=ab[cuda.threadIdx.x]
    if(cuda.threadIdx.x==cuda.blockDim.x-1):
        s[cuda.blockIdx.x]=ab[cuda.threadIdx.x]

@cuda.jit('void(int32[:],int32[:],int32)')
def pref_sum_update(b,s,length):
    tid=(cuda.blockIdx.x+1)*cuda.blockDim.x+cuda.threadIdx.x
    if tid<length:
        b[tid]+=s[cuda.blockIdx.x]

@cuda.jit('void(int32[:], int32[:], int32[:], int32)')
def map_non_zeros(a,prefix_sum,nz,length):
    tid=cuda.blockIdx.x*cuda.blockDim.x+cuda.threadIdx.x
    if tid<length:
        input_value=a[tid]
        if input_value!=0:
            index=prefix_sum[tid]
            nz[index-1]=input_value

def pref_sum(a,asum,nzm):
    block=BLOCK_SIZE
    length=a.shape[0]
    grid=int((length+block-1)/block)
    bs=cuda.device_array(shape=(grid),dtype=np.int32)
    prefix_sum_nzmask_block[grid,block](a,asum,bs,nzm,length)
    if grid>1:
        bssum=cuda.device_array(shape=(grid),dtype=np.int32)
        pref_sum(bs,bssum,0)
        pref_sum_update[grid-1,block](asum,bssum,length)

def get_non_zeros(a):
    ac=np.ascontiguousarray(a)
    ad=cuda.to_device(ac)
    bd=cuda.device_array_like(ad)
    pref_sum(ad,bd,int(1))
    non_zero_count=int(bd[bd.shape[0]-1])
    non_zeros=cuda.device_array(shape=(non_zero_count),dtype=np.int32)
    block=BLOCK_SIZE
    length=a.shape[0]
    grid=int((length+block-1)/block)
    map_non_zeros[grid,block](ad,bd,non_zeros,length)
    return non_zeros.copy_to_host()

@cuda.jit
def get_bound_cuda2(tmp_img,bound_info):
    r,c=cuda.grid(2)
    # last=0,1,2,3 for n,e,s,w respectively
    if r%3==0 and c%3==0 and tmp_img[r][c]!=500:
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
        bound_info[min_i][0]=min_i
        bound_info[min_i][1]=n
        #cuda.atomic.max(bound_max_d[min_i],0,d_tmp_max)


@cuda.jit
def get_bound_cuda(tmp_img,bound_array_d,max_i):
    r,c=cuda.grid(2)
    # last=0,1,2,3 for n,e,s,w respectively
    if r%3==0 and c%3==0 and tmp_img[r][c]!=500:
        y=r
        x=c
        i_=int(y/3)*int(tmp_img.shape[1]/3)+int(x/3)
        if i_>max_i[0]:
            max_i[0]=i_
        color=tmp_img[r][c]
        n=0
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
            bound_array_d[i_][n]=r*tmp_img.shape[1]+c
            n+=1
            if r==y and c==x:
                break
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
            else:
                break

@cuda.jit
def fence_image(img_array,img_fenced_d):
    r,c=cuda.grid(2)
    if r==0 or c==0 or r==img_array.shape[0]-1 or c==img_array.shape[1]-1:
        img_fenced_d[r][c]=500

def get_bound(img):
    blob_dict={}
    img_array=img
    threadsperblock=(1,1)
    blockspergrid_x=math.ceil(img_array.shape[0]/threadsperblock[0])
    blockspergrid_y=math.ceil(img_array.shape[1]/threadsperblock[1])
    blockspergrid=(blockspergrid_x,blockspergrid_y)
    n=0
    while 1:
        start_time=time.time()
        pos_h=np.full((1,2),(-1,-1))
        pos_d=cuda.to_device(pos_h)
        img_array_d=cuda.to_device(img_array)
        get_color_index_cuda[blockspergrid,threadsperblock](img_array_d,pos_d)
        cuda.synchronize()
        i=pos_d.copy_to_host()

        #print("\tfinished indexing in",time.time()-start_time)
        start_time=time.time()
        if i[0][0]==-1:
            return blob_dict
        #r,c=i_to_p(i[0],img_array.shape)
        r,c=i[0]
        color=img_array[r][c]
        if color in blob_dict:
            blob_dict[color].append(list())
        else:
            blob_dict[color]=[[]]
        start_time=time.time()
        while 1:
            img_array[r][c]=500
            blob_dict[color][-1].append(p_to_i((r,c),img_array.shape))
            n+=1
            if img_array[r-1][c]==color:
                r-=1
            elif img_array[r][c+1]==color:
                c+=1
            elif img_array[r+1][c]==color:
                r+=1
            elif img_array[r][c-1]==color:
                c-=1
            else:
                break
        print("\tFound bound of size",len(blob_dict[color][-1]),"in",time.time()-start_time)
    return blob_dict

@cuda.jit
def find_max(arr_,max_):
    i=cuda.grid(1)
    if arr_[i]>max_[0]:
        max_[0]=arr_[i]

@cuda.jit
def get_color_index_cuda(img_array,pos):
    r,c=cuda.grid(2)
    if r>0 and r<img_array.shape[0]-1 and c>0 and c<img_array.shape[1]-1:
        if img_array[r][c]!=500:
            #pos[0]=img_array.shape[1]*r+c
            pos[0]=(r,c)

@cuda.jit
def compute_avg_img_cuda(img,avg_img):
    r,c=cuda.grid(2)
    if r>0 and r<img.shape[0]-1 and c>0 and c<img.shape[1]-1:
        avg=(img[r-1][c-1]+img[r-1][c]+img[r-1][c+1]+img[r][c-1]+img[r][c]+img[r][c+1]+img[r+1][c-1]+img[r+1][c]+img[r+1][c+1])/9.0
        avg_img[r][c]=avg

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
    


def convert_color(img,output_format):
    converted_img=cudaAllocMapped(width=img.width,height=img.height,
            format=output_format)
    cudaConvertColor(img,converted_img)
    return converted_img

def condition_img_nv(img_array):
    img_cond=np.ones([img_array.shape[0]*3,img_array.shape[1]])
    for r in range(img_array.shape[0]):
        for i in range(3):
            img_cond[r*3+i]=img_array[r]
    img_array=img_cond.transpose()
    img_cond=np.ones([img_array.shape[0]*3,img_array.shape[1]])
    for r in range(img_array.shape[0]):
        for i in range(3):
            img_cond[r*3+i]=img_array[r]
    img_cond=img_cond.transpose()
    return img_cond


def draw_pixels_cuda(pixels,i,img):
    draw_pixels_cuda_[pixels.shape[0],1](pixels,i,img)
    cuda.synchronize()

@cuda.jit
def draw_pixels_cuda_(pixels,i,img):
    cc=cuda.grid(1)
    if cc<pixels.shape[0]:
        r=int(pixels[cc]/img.shape[1])
        c=pixels[cc]%img.shape[1]
        img[r][c]=i
 
def draw_pixels_from_indices_cuda(indices,pixels,i,img):
    draw_pixels_from_indices_cuda_[indices.shape[0],1](indices,pixels,i,img)
    cuda.synchronize()

@cuda.jit
def draw_pixels_from_indices_cuda_(indices,pixels,i,img):
    cc=cuda.grid(1)
    if cc<len(indices):
        r=int(pixels[indices[cc]]/img.shape[1])
        c=pixels[indices[cc]]%img.shape[1]
        img[r][c]=i

def write_image(fname,image):
    Image.fromarray(image).save(fname)

def open_image(fname):
    """returns the image as numpy array"""
    return np.array(Image.open(fname).convert('L'))

if __name__=="__main__":
    main()
