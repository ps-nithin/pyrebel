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
import argparse
import time
from jetson_utils import videoSource, videoOutput, Log
from jetson_utils import cudaAllocMapped,cudaConvertColor
from jetson_utils import cudaToNumpy,cudaDeviceSynchronize,cudaFromNumpy
from numba import cuda
import numpy as np
import cmath

# create video sources & outputs
input = videoSource("csi://0", options={'width':320,'height':240,'framerate':30,'flipMethod':'rotate-270'})
output = videoOutput("", argv=sys.argv)

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
        #done=new_image(scaled_shape)
        #write_image(out_loc,done)
        #img_boundary_h=img_boundary_d.copy_to_host()
        #print(img_boundary_h)
        img_boundary_cuda=cudaFromNumpy(out_image_h)
        img_boundary_cuda_rgb=convert_color(img_boundary_cuda,'rgb8')
        # render the image
        output.Render(img_boundary_cuda_rgb)
        # exit on input/output EOS
        #if not input.IsStreaming() or not output.IsStreaming():
        #    break
        n+=1
        print("Finished in total of",time.time()-init_time,"seconds at",float(1/(time.time()-init_time)),"fps count=",n)
        #im=Image.fromarray(out_image_h).convert('L')
        #im.save("cuda_scaled.png")

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
        threshold=sqrt(float(pow(y2-y,2)+pow(x2-x,2)))/16
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

from PIL import Image
import numpy as np
import time
import sys
from math import sqrt
import math
import json
from operator import sub
import matplotlib.pyplot as plot


http_loc="/data/data/com.termux/files/usr/share/apache2/default-site/htdocs/done.png"
out_loc="done.png"

unique_counter=0
done=0

def init_run(fname):
    """
        Identify points of temporary balance. 
        Big changes have big influence.
        Small changes have little influence.
        A balance is restored when there occurs a series of ups followed by 
        an equal no of downs or vice versa.
         color2:    ...  }
    """
    img=open_image(fname)
    img_cond=condition_img(img)
    sort_dict_and_write_json(img_cond,fname)   
    return img_cond.shape

def main_old():
    

    """ 
    REVERSE MAPPING 
    SYMBOL : MEANING OF THE WORD
    MEANING : SYMBOL
    OF : SYMBOL
    THE : SYMBOL
    WORD : SYMBOL

    IT SHOULD BE POSSIBILE TO EXPRESS AND RECOGNIZE A PATTERN WITH EXCESS
    AMOUNT OF DATA SUCH THAT THE BLUEPRINT REMAINS THE SAME.

    THE EXCESS DATA IS REDUCED TO A BASIC UNDERSTANDING OF THE PATTERN.


    LINE(SLOPE,LENGTH,POSITION)
    Lines of similar slope, length, or positon are grouped.

    REPRESENT GROUP OF PIXELS WITH A UNIQUE IDENTIFIER/SYMBOL.
    POINTS OF CHANGE IN DIRECTION AND HOW LONG IN THAT DIRECTION.
    
    vline and hline are primary symbols.
    size of hline and vline relative to the size of the canvas
    influences whether it is perceived as a line or a point.
    
    displacement is considered in visual perception not distance.
    
    imagine that every word is a selector of subsets from a set.
    for eq. word 'top' divides the set into to two and selects
    the upper data.

    nothing is perfect. so is a sloping line. vline and hline can
    be perfect theoretically but a sloping line is always a 
    combination of vline and hline. every symbol is expressed as 
    a confidence level of its true definition.
    
    every symbol has a blueprint
    blueprints are made of existing symbols
    so when the symbols VERTICAL LINE and HORIZONTAL LINE are close to 
    perpendicular. the data speaks what it is. its features speaks out loud to
    the world what it is. then occurs coincidence of a set of features which 
    is recognized as objects.
    
    features are pushed into memories.
    
    EVIDENCE IN NO SPECIFIC ORDER.
    CLOSED FIGURE IS AN OBJECT WHERE THE ORDER IS IRRELEVANT.

    POWER OF OBSERVING EVIDENCES IN THE INPUT
    THAT GIVE RISE THE QUESTION, WHAT IS AN EVIDENCE?
    IN SIMPLE TERMS AN EVIDENCE IS A PATTERN
    AND PATTERN IS INPUT THAT FOLLOWS DEFINITE SIGNATURE.
    MEANS, SOMETHING THAT REPEATS WITH DEFINITE CHANGES.
    THE DEEPEST QUESTION IS - WHAT IS AN EVIDENCE?
    WHY DO WE RECOGNIZE EVIDENCE? 
    EVERY PIXEL IN AN IMAGE IS TAGGED TO AN EVIDENCE.
    COLOR IS AN EVIDENCE. POSITION IS AN EVIDENCE.
    GROUP OF PIXELS ARE TAGGED TO A SINGLE EVIDENCE.
    RICHNESS OF EXPERIENCE IS A CONSEQUENCE OF THE RICHNESS
    OF EVIDENCE TAGGED.


    RESOLUTION OF PERCEPTION REMAINS THE SAME. 
    THAT IS, IF THE INPUT IS 500 POINTS AND THE RESOLUTION 
    OF BRAIN ANALYZING THE INPUT IS 10 THEN THE DETAILS ARE
    50 POINTS APART. THUS, IF THE INPUT IS 100 POINTS THEN
    THE DETAILS ARE 100/10 THAT IS 10 POINTS APART.
    THE RESOLVING POWER OF THE BRAIN REMAINS THE SAME REGARDLESS OF 
    THE INPUT COMPLEXITY.

    Effectively, sensory data stream is nothing but a series of valleys and
    curves, big and small.

    A valley or curve of 3 points is characterized by a maxima of perpendicular
    distance of the middle point.

    Mind takes samples from the sensory input stream at equidistant points.

    Max per dist between two points of zero per dist.

    Complexity ratio :  a measure of the complexity between any two points in 
                        the input.
    """
    #fname=sys.argv[1]
    fname="aaa.png"
    start_time=time.time()
    #init_run(fname)
    blob_dict=read_json(fname)
    shape=read_shape(fname)
    print("shape: ",shape)
    #blob_dict_summary(blob_dict)
    blob1=blob_dict[0][0]
    blob2=blob_dict[0][2]

    plot_one="plot.jpg"
    plot_two="plot2.jpg"
    global done
    done=new_image(shape)
    #img=open_image(fname)
    #color=img[50][530]
    #blob=blob_dict[color][0][::]
    draw_pixels_i(blob1,200,done)
    draw_pixels_i(blob2,200,done)
    print("len(blob1)=",len(blob1))
    print("len(blob2)=",len(blob2))
    #dist=int(sys.argv[1])
    dist1=7
    dist2=10
    n_eq_pixels=1
    
    
    pd_avg1=get_per_dist_all(n_eq_pixels,dist1,blob1,shape,done)
    opt_softness1=recursive_average(pd_avg1,500)
    pd_avg1=compute_average_ntimes(opt_softness1,pd_avg1)
    pat1=n_inflections(pd_avg1)
    pd_avg1=start_from_nth(pat1[0],pd_avg1)
    pat1=n_patterns(pd_avg1)
    disc_pd1=make_discrete(pd_avg1)

    """
    pd_avg2=get_per_dist_all(n_eq_pixels,dist2,blob2,shape,done)
    #opt_softness2=recursive_average(pd_avg2,500)
    pd_avg2=compute_average_ntimes(100,pd_avg2)
    pat2=n_inflections(pd_avg2)
    pd_avg2=start_from_nth(pat2[0],pd_avg2)
    pat2=n_patterns(pd_avg2)
    disc_pd2=make_discrete(pd_avg2)
    """
    pd_pos1=list(range(0,len(pd_avg1)))
    plot.figure(figsize=(20,5))
    plot.bar(pd_pos1[:],disc_pd1[:])
    plot.plot(pat1,[0 for i in range(0,len(pat1))],marker='*',color='red')
    plot.savefig(plot_one)
    plot.clf()
    pd_pos1=list(range(0,len(pd_avg1)))
    plot.bar(pd_pos1[:],disc_pd1[:])
    plot.plot(pat1,[0 for i in range(0,len(pat1))],marker='*',color='red')
    plot.savefig(plot_two)
    print("finished in",time.time()-start_time,"seconds.")
    write_image(out_loc,done)

def make_discrete(pd_list):
    pat=n_patterns(pd_list)
    pat.append(len(pd_list)-1)
    disc_pd_list=list()
    i_pat=0
    for i in range(0,len(pd_list)):
        if i==pat[i_pat+1]:
            i_pat+=1
        disc_pd_list.append(pd_list[pat[i_pat]])
 
    return disc_pd_list

def find_peak(pat,pd_list_ip):
    pd_avg=list()
    for i in pd_list_ip:
        if abs(i)<0.01:
            pd_avg.append(0)
        else:
            pd_avg.append(i)
    pat_new=list()
    for i in range(1,len(pat)-1):
        if abs(pd_avg[pat[i-1]])<abs(pd_avg[pat[i]]) and abs(pd_avg[pat[i]])>abs(pd_avg[pat[i+1]]):
            pat_new.append(pat[i])
    return pat_new

def find_valley(pd_list_ip):
    pd_avg=list()
    for i in pd_list_ip:
        if abs(i)<0.01:
            pd_avg.append(0)
        else:
            pd_avg.append(i)
    valley=list()
    for i in range(1,len(pd_avg)-1):
        if abs(pd_avg[i-1])>abs(pd_avg[i]) and abs(pd_avg[i])<abs(pd_avg[i+1]):
            valley.append(i)
    return valley

def find_zero(pd_list_ip):
    pd_list=list()
    for i in pd_list_ip:
        if abs(i)<0.01:
            pd_list.append(0)
        else:
            pd_list.append(i)
    pat=list()
    for i in range(1,len(pd_list)-1):
        prev_pd=pd_list[i-1]
        cur_pd=pd_list[i]
        next_pd=pd_list[i+1]
        if prev_pd!=0 and cur_pd==0:
            pat.append(i)
        elif prev_pd==0 and cur_pd!=0:
            pat.append(i)
        elif prev_pd<cur_pd and next_pd<cur_pd:
            pat.append(i)
        elif prev_pd>cur_pd and next_pd>cur_pd:
            pat.append(i)
    return pat


def find_curve(pd_list_ip):
    pd_avg=list()
    for i in pd_list_ip:
        if abs(i)<0.01:
            pd_avg.append(0)
        else:
            pd_avg.append(i)
 
    curve=list()
    for i in range(1,len(pd_avg)-1):
        if abs(pd_avg[i-1])<abs(pd_avg[i]) and abs(pd_avg[i])>abs(pd_avg[i+1]):
            curve.append(i)
    return curve


def divide_list(pat,blob):
    blob_pat=list()

    for i in range(1,len(pat)):
        blob_pat.append(list())
        for j in range(pat[i-1],pat[i]):
            blob_pat[i-1].append(blob[j])
    blob_pat.append(list())
    for i in range(pat[-1],len(blob)):
        blob_pat[-1].append(blob[i])
    for i in range(0,pat[0]):
        blob_pat[-1].append(blob[i])
    return blob_pat


def n_inflections(pd_list_ip):
    pat=list()
    pd_list=pd_list_ip
    """
    for i in pd_list_ip:
        if abs(i)<0.01:
            pd_list.append(0)
        else:
            pd_list.append(i)
    """
    for i in range(1,len(pd_list)):
        prev_pd=pd_list[i-1]
        cur_pd=pd_list[i]
        if prev_pd!=0 and cur_pd==0:
            pat.append(i)
        elif prev_pd==0 and cur_pd!=0:
            pat.append(i)
        elif prev_pd<0 and cur_pd>0:
            pat.append(i)
        elif prev_pd>0 and cur_pd<0:
            pat.append(i)
    return pat

def n_patterns(pd_list_ip):
    pat=list()
    pd_list=list()
    for i in pd_list_ip:
        if abs(i)<0.01:
            pd_list.append(0)
        else:
            pd_list.append(i)

    prev_pd=pd_list[-1]
    cur_pd=pd_list[0]
    next_pd=pd_list[1]
    if prev_pd!=0 and cur_pd==0:
        pat.append(0)
    elif prev_pd==0 and cur_pd!=0:
        pat.append(0)
    elif prev_pd>0 and cur_pd<0:
        pat.append(0)
    elif prev_pd<0 and cur_pd>0:
        pat.append(0)
    elif prev_pd<cur_pd and next_pd<cur_pd:
        pat.append(0)
    elif prev_pd>cur_pd and next_pd>cur_pd:
        pat.append(0)

    for i in range(1,len(pd_list)-1):
        prev_pd=pd_list[i-1]
        cur_pd=pd_list[i]
        next_pd=pd_list[i+1]
        if prev_pd!=0 and cur_pd==0:
            pat.append(i)
        elif prev_pd==0 and cur_pd!=0:
            pat.append(i)
        elif prev_pd<cur_pd and next_pd<cur_pd:
            pat.append(i)
        elif prev_pd>cur_pd and next_pd>cur_pd:
            pat.append(i)
        elif prev_pd>0 and cur_pd<0:
            pat.append(i)
        elif prev_pd<0 and cur_pd>0:
            pat.append(i)
    prev_pd=pd_list[-2]
    cur_pd=pd_list[-1]
    next_pd=pd_list[0]
    i=len(pd_list)-1
    if prev_pd!=0 and cur_pd==0:
        pat.append(i)
    elif prev_pd==0 and cur_pd!=0:
        pat.append(i)
    elif prev_pd>0 and cur_pd<0:
        pat.append(i)
    elif prev_pd<0 and cur_pd>0:
        pat.append(i)
    elif prev_pd<cur_pd and next_pd<cur_pd:
        pat.append(i)
    elif prev_pd>cur_pd and next_pd>cur_pd:
        pat.append(i)

    return sorted(list(set(pat)))

def recursive_pattern(pd_list,n_recursion):
    pd_avg=pd_list
    n=0
    npat_count={}
    while 1:
        pd_avg=compute_average_segment(pd_avg)
        pat=n_patterns(pd_avg)
        npat=len(pat)
        if npat in npat_count:
            npat_count[npat]["count"]+=1
        else:
            npat_count[npat]={"count":1,"avg_start":n+1}
        if n==n_recursion or npat==2:
            break
        n+=1
    npat_count_max=0
    max_key=-1
    for key,val in npat_count.items():
        if val["count"]>npat_count_max:
            npat_count_max=val["count"]
            max_key=key
    print("optimal_pattern_count=",max_key)
    print("optimal_softness=",npat_count[max_key]["avg_start"])
    print("")
    return npat_count[max_key]["avg_start"]


def recursive_average(pd_list,n_recursion):
    pd_avg=pd_list
    n=0
    npat_count={}
    while 1:
        if n==n_recursion:
            break
        pd_avg=compute_average(pd_avg)
        pat=n_inflections(pd_avg)
        npat=len(pat)
        if npat in npat_count:
            npat_count[npat]["count"]+=1
        else:
            npat_count[npat]={"count":1,"avg_start":n+1}
        n+=1
    npat_count_max=0
    max_key=-1
    for key,val in npat_count.items():
        if val["count"]>npat_count_max:
            npat_count_max=val["count"]
            max_key=key
    print("optimal_inflection_count=",max_key)
    print("optimal_softness=",npat_count[max_key]["avg_start"])
    return npat_count[max_key]["avg_start"]


def compute_average(pd_list):
    pd_avg=list()
    pd_avg.append((pd_list[-1]+pd_list[0]+pd_list[1])/3)
    for i in range(1,len(pd_list)-1,1):
        pd_avg.append((pd_list[i-1]+pd_list[i]+pd_list[i+1])/3)
    pd_avg.append((pd_list[-2]+pd_list[-1]+pd_list[0])/3)
    return pd_avg

def compute_average_segment(pd_list):
    pd_avg=list()
    pd_avg.append((pd_list[0]+pd_list[1])/3)
    for i in range(1,len(pd_list)-1,1):
        pd_avg.append((pd_list[i-1]+pd_list[i]+pd_list[i+1])/3)
    pd_avg.append((pd_list[-2]+pd_list[-1]+0)/3)
    return pd_avg

def compute_average_segment_ntimes(n,pd_list):
    i=0
    pd_avg=pd_list
    while 1:
        if i==n:
            break
        pd_avg=compute_average_segment(pd_avg)
        i+=1
    return pd_avg


def compute_average_ntimes(n,pd_list):
    i=0
    pd_avg=pd_list
    while 1:
        if i==n:
            break
        pd_avg=compute_average(pd_avg)
        i+=1
    return pd_avg


def get_islands(pd_list):
    cur_min=0
    cur_min_pos=0
    min_list=list()
    for i in range(1,len(pd_list)):
        cur_pd=pd_list[i]
        prev_pd=pd_list[i-1]
        if (cur_pd<=0 and prev_pd<=0):
            if cur_pd<cur_min:
                cur_min=cur_pd
                cur_min_pos=i
        else:
            cur_min=0
            if len(min_list)==0 or min_list[-1]!=cur_min_pos:
                min_list.append(cur_min_pos)
    return min_list


def get_per_dist_all(n,max_dist,blob,shape,done):
    pd_list=list()
    for dist in range(max_dist,max_dist+1):
        print("dist=",dist)
        for s in range(0,len(blob)):
            pd=get_per_dist_both(s,n,dist,blob,shape)
            pd_list.append(pd)
    #pd_sorted=sorted(pd_list,key=lambda x:x[2],reverse=True)
    return pd_list

def get_per_dist_both(start,n,dist,blob,shape):
    indices=get_n_equidistant_pixels_both(start,n,dist,blob,shape)
    pxls=get_pixels_from_index(indices,blob)
    pd=per_dist_i(pxls[-1],pxls[-2],pxls[0],shape)
    return pd

def get_n_equidistant_pixels_both(s,n,dist,blob,shape):
    pixels_forward=get_n_equidistant_pixels_forward(s,n+1,dist,blob,shape)
    pixels_backward=get_n_equidistant_pixels_backward(s,n+1,dist,blob,shape)
    pixels_both=list()
    pixels_both.append(pixels_forward[0])
    for i in range(1,len(pixels_forward)):
        pixels_both.append(pixels_backward[i])
        pixels_both.append(pixels_forward[i])
    return pixels_both

def get_cycles(pd_list):
    pd_list_sorted=start_from_min(pd_list)
    min_diff=9999
    min_diff_pos=0
    out=list()
    out.append(0)
    n=0
    while 1:
        for pos in range(out[-1]+1,len(pd_list)-5):
            diff=pd_list_sorted[pos]-pd_list_sorted[out[-1]]
            if diff<min_diff:
                min_diff=diff
                min_diff_pos=pos
        out.append(min_diff_pos)
        min_diff=9999
        if n==40:
            break
        n+=1
    return out

def get_eq_pixels(dist,blob,shape):
    eq_pixels=list()
    start=0
    eq_pixels.append(start)
    last_found=eq_pixels[-1]
    next_pixel=next_i(last_found,blob)
    while 1:
        dist_temp=int(dist_i(blob[last_found],blob[next_pixel],shape))
        dist_diff=abs(dist_temp-dist)
        if dist_diff==0:
            if next_pixel>start and next_pixel<step(start,blob,dist-1):
                break
            eq_pixels.append(next_pixel)
            last_found=eq_pixels[-1]
        next_pixel=next_i(next_pixel,blob)
        if next_pixel==start:
            break
    return eq_pixels


def start_from_max(pd_list):
    pd_max=0
    pd_max_pos=0
    for i in range(0,len(pd_list)):
        if pd_list[i]>pd_max:
            pd_max=pd_list[i]
            pd_max_pos=i
    pd_list_new=list()
    for i in range(pd_max_pos,len(pd_list)):
        pd_list_new.append(pd_list[i])
    for i in range(0,pd_max_pos):
        pd_list_new.append(pd_list[i])
    return pd_list_new

def start_from_min(pd_list):
    pd_min=99999
    pd_min_pos=0
    for i in range(0,len(pd_list)):
        if pd_list[i]<pd_min:
            pd_min=pd_list[i]
            pd_min_pos=i
    pd_list_new=list()
    for i in range(pd_min_pos,len(pd_list)):
        pd_list_new.append(pd_list[i])
    for i in range(0,pd_min_pos):
        pd_list_new.append(pd_list[i])
    return pd_list_new

def start_from_nth(n,ip_list):
    list_new=list()
    for i in range(n,len(ip_list)):
        list_new.append(ip_list[i])
    for i in range(0,n):
        list_new.append(ip_list[i])
    return list_new

def get_zero_pos(pd_list):
    for i in range(0,len(pd_list)):
        if pd_list[i]==0:
            return i
    return 0

def get_min_pos(pd_list):
    pd_min=99999
    pd_min_pos=0
    for i in range(0,len(pd_list)):
        if pd_list[i]<pd_min:
            pd_min=pd_list[i]
            pd_min_pos=i
    return pd_min_pos
 

def start_from_min_pd(pd_pos_list):
    pd_min=999999
    pd_min_pos=0
    for i in range(0,len(pd_pos_list)):
        if pd_pos_list[i][2]<pd_min:
            pd_min=pd_pos_list[i][2]
            pd_min_pos=i
    print(pd_min,pd_min_pos)
    pd_pos_list_new=list()
    for i in range(pd_min_pos,len(pd_pos_list)):
        pd_pos_list_new.append(pd_pos_list[i])
    for i in range(0,pd_min_pos):
        pd_pos_list_new.append(pd_pos_list[i])
    return pd_pos_list_new


def get_inflections(n,dist,blob,shape):
    d_pre=get_inflection_displacement(0,n,dist,blob,shape)
    inflections=list()
    for s in range(1,len(blob)):
        d=get_inflection_displacement(s,n,dist,blob,shape)
        if (d-d_pre)>3:
            indices=get_n_equidistant_pixels(s-1,n,dist,blob,shape)
            inflections.append(blob[indices[-1]])
        d_pre=d
    return inflections

 
def get_inflection_displacement(s,n,eq_dist,blob,shape):
    dist=0
    indices=get_n_equidistant_pixels(s,n,eq_dist,blob,shape)
    i=indices[int(len(indices)/2)]
    while 1:
        dist+=1
        i=next_i(i,blob)
        if i==indices[-1]:
            break
    return dist

def get_max_per_dist3(n,max_dist,blob,shape,done):
    pd_list=list()
    for dist in range(max_dist,max_dist+1):
        print("dist=",dist)
        for s in range(0,len(blob)):
            pd=get_per_dist(s,n,dist,blob,shape)
            pd_list.append([s,dist,pd])
    #pd_sorted=sorted(pd_list,key=lambda x:x[2],reverse=True)
    return pd_list

def get_max_per_dist2(n,max_dist,blob,shape,done):
    pd_list=list()
    pd_unique=list()
    for dist in range(20,max_dist):
        print("dist=",dist)
        for s in range(0,len(blob)):
            pd=get_per_dist(s,n,dist,blob,shape)
            pd_list.append([s,dist,pd])
    pd_sorted=sorted(pd_list,key=lambda x:x[2],reverse=True)
    print("len(pd_sorted)=",len(pd_sorted))
    pd_unique.append(pd_sorted[0])
    counter=0
    _is_child_=False
    for pd in pd_sorted:
        counter+=1
        print("len(pd_sorted)=",len(pd_sorted),"counter=",counter,"len(pd_unique)=",len(pd_unique))
        for pd_unique_ in pd_unique:
            if is_child2(n,pd,pd_unique_,blob,shape):
                _is_child_=True
                break
            _is_child_=False
        if not _is_child_:
            pd_unique.append(pd)
            _is_child_=False
        if counter==10000:
            break
    print("len(pd_unique)=",len(pd_unique))
    return pd_unique

def is_child2(n,pd_ip,pd_unique,blob,shape):
    indices_ip=get_n_equidistant_pixels(pd_ip[0],n,pd_ip[1],blob,shape)
    middle_ip=blob[indices_ip[int(n/2)]]
    indices_unique=get_n_equidistant_pixels(pd_unique[0],n,pd_unique[1],blob,shape)
    unique_pixels=get_pixels_from(indices_unique[0],indices_unique[-1],blob)
    if middle_ip in unique_pixels:
        return True
    else:
        return False


def get_per_dist(start,n,dist,blob,shape):
    indices=get_n_equidistant_pixels_forward(start,n,dist,blob,shape)
    middle_pos=indices[int(n/2)]
    pd=per_dist_i(blob[indices[0]],blob[indices[-1]],blob[middle_pos],shape)
    return pd

def get_all_min_si_eq(n,blob,shape):
    #dist=int(farthest_pixels(blob,shape)[0]/4)
    dist=50
    while 1:
        print("dist=",dist)
        si_min,si_min_pos=get_best_si_eq(n,dist,blob,shape)
        if si_min==0:
            print("dist=",dist,"si_min=",si_min,"len(si_min_pos)=",len(si_min_pos))
            break
        dist-=10
        if dist<10:
            break
    
    return [si_min_pos,dist]

def get_best_si_eq(n,dist,blob,shape):
    si_min=99999999
    si_min_pos=list()
    for i in range(0,len(blob)):
        eq_pixels_index=get_n_equidistant_pixels(i,n,dist,blob,shape)
        eq_pixels=get_pixels_from_index(eq_pixels_index,blob)
        if not on_same_side(0,n-1,eq_pixels,shape):
            continue
        si_temp=get_symmetry_index_wrap(eq_pixels,shape)
        if si_temp<si_min:
            si_min=si_temp
            si_min_pos.clear()
            si_min_pos.append(i)
        if si_temp==si_min:
            if i not in si_min_pos:
                si_min_pos.append(i)
    return [si_min,si_min_pos]

def get_pixels_from_index(indices,blob):
    pixels=list()
    for i in indices:
        pixels.append(blob[i])
    return pixels

def get_symmetry_index_wrap(eq_pixels,shape):
    return get_symmetry_index(0,len(eq_pixels)-1,eq_pixels,shape)

def get_n_equidistant_pixels_forward(start,n,dist,blob,shape):
    eq_pixels=list()
    eq_pixels.append(start)
    last_found=eq_pixels[-1]
    next_pixel=next_i(last_found,blob)
    found=1
    while 1:
        if found==n:
            break
        dist_temp=int(dist_i(blob[last_found],blob[next_pixel],shape))
        dist_diff=abs(dist_temp-dist)
        if dist_diff==0:
            eq_pixels.append(next_pixel)
            last_found=eq_pixels[-1]
            found+=1
        next_pixel=next_i(next_pixel,blob)
        if next_pixel==start:
            break
    return eq_pixels

def get_n_equidistant_pixels_backward(start,n,dist,blob,shape):
    eq_pixels=list()
    eq_pixels.append(start)
    last_found=eq_pixels[-1]
    next_pixel=prev_i(last_found,blob)
    found=1
    while 1:
        if found==n:
            break
        dist_temp=int(dist_i(blob[last_found],blob[next_pixel],shape))
        dist_diff=abs(dist_temp-dist)
        if dist_diff==0:
            eq_pixels.append(next_pixel)
            last_found=eq_pixels[-1]
            found+=1
        next_pixel=prev_i(next_pixel,blob)
        if next_pixel==start:
            break
    return eq_pixels


def farthest_pixels(blob,shape):
    max_disp=0
    index_i=0
    index_j=0
    for i in range(0,len(blob)):
        for j in range(0,len(blob)):
            temp_disp=dist_i(blob[i],blob[j],shape)
            if temp_disp>max_disp:
                max_disp=temp_disp
                index_i=i
                index_j=j
    return [max_disp,index_i,index_j]

def combine_corner_angle_size(angle_array,size_array):
    corner_array=list()
    for i in range(0,len(angle_array)):
        corner_array.append([angle_array[i][0],angle_array[i][1],size_array[i][1]])
    return corner_array


def get_smallest_corner_angle(corners):
    corner_angle_sorted=sorted(corners,key=lambda x: x[1],reverse=False)
    smallest_angle=corner_angle_sorted[0][1]
    smallest_corners=list()
    for c in corner_angle_sorted:
        if c[1]==smallest_angle:
            smallest_corners.append(c)
        else:
            break
    return smallest_corners

def get_smallest_corner_size(corners):
    corner_size_sorted=sorted(corners,key=lambda x: x[2],reverse=False)
    smallest_size=corner_size_sorted[0][2]
    smallest_corners=list()
    for c in corner_size_sorted:
        if c[2]==smallest_size:
            smallest_corners.append(c)
        else:
            break
    return smallest_corners



def get_corner_angle(blob,shape):
    corner_angle=list()
    l=angle_i(blob[-1],blob[0],shape)
    r=angle_i(blob[0],blob[1],shape)
    lr_diff=abs(angle_diff(l,r))
    corner_angle.append([blob[0],lr_diff])
    for i in range(1,len(blob)-1):
        l=angle_i(blob[i-1],blob[i],shape)
        r=angle_i(blob[i],blob[i+1],shape)
        lr_diff=abs(angle_diff(l,r))
        corner_angle.append([blob[i],lr_diff])
    l=angle_i(blob[-2],blob[-1],shape)
    r=angle_i(blob[-1],blob[0],shape)
    lr_diff=abs(angle_diff(l,r))
    corner_angle.append([blob[-1],lr_diff])
    return corner_angle


def get_corner_size(blob,shape):
    corner_size=list()
    size_l=dist_i(blob[-1],blob[0],shape)
    size_r=dist_i(blob[0],blob[1],shape)
    corner_size.append([blob[0],size_l+size_r])
 
    for i in range(1,len(blob)-1):
        size_l=dist_i(blob[i-1],blob[i],shape)
        size_r=dist_i(blob[i],blob[i+1],shape)
        corner_size.append([blob[i],size_l+size_r])
    size_l=dist_i(blob[-2],blob[-1],shape)
    size_r=dist_i(blob[-1],blob[0],shape)
    corner_size.append([blob[-1],size_l+size_r])
    return corner_size



def get_ups(blob,shape,recursion):
    ups=blob
    i=0
    while 1:
        ups=get_outer_corner(ups,shape)
        i+=1
        if i==recursion:
            break
    print("len(ups)=",len(ups))
    return ups

def get_downs(blob,shape,recursion):
    downs=blob
    i=0
    while 1:
        downs=get_inner_corner(downs,shape)
        i+=1
        if i==recursion:
            break
    print("len(downs)=",len(downs))
    return downs


def is_parent_blob(ip_blob,blob):
    for i in blob:
        if i not in ip_blob:
            return 0
    return 1
def is_child_blob(ip_blob,blob):
    for i in ip_blob:
        if i not in blob:
            return 0
    return 1

def is_child(sl_ip,sl,blob):
    """
    checks if [s_ip,l_ip] is child of [s,l]
    """
    ip_blob=get_pixels_from(sl_ip[0],
            step(sl_ip[0],blob,sl_ip[1]),blob)
    blob=get_pixels_from(sl[0],
            step(sl[0],blob,sl[1]),blob)   
    return is_child_blob(ip_blob,blob)

def is_parent(sl_ip,sl,blob):
    ip_blob=get_pixels_from(sl_ip[0],
            step(sl_ip[0],blob,sl_ip[1]),blob)
    blob=get_pixels_from(sl[0],
            step(sl[0],blob,sl[1]),blob)   
    return is_parent_blob(ip_blob,blob)

def has_pair(s,e,blob,shape):
    np=list()
    i=s
    last=0
    if not on_same_side(s,e,blob,shape):
        return False
    while(1):
        i=next_i(i,blob)
        if i==e:
            np.append(-1)
            break

        pd=abs(per_dist_i(blob[s],blob[e],blob[i],shape))
        if pd>last:
            np.append(1)
        elif pd<last:
            np.append(-1)
        else:
            return False
            np.append(0)
        last=pd
    np2=list()
    np2.append(np[0])
    for i in range(1,len(np)):
        if np2[-1]!=np[i]:
            np2.append(np[i])
    if np2.count(1)>1 or np2.count(-1)>1 or np2.count(0)>1:
        return False
    return True



def get_pair_full_nonc(pixel_group,shape):
    res=list()
    for i in range(3,len(pixel_group),1):
        res1=get_pair_blob_nonc(i,pixel_group,shape)
        for r in res1:
            res.append(r)
    
    temp=list()
    found_child=False
    for r in res:
        found_child=False
        for rr in res:
            if is_child(r,rr,pixel_group) and r!=rr:
                found_child=True
                break
        if not found_child:
            temp.append(r)
            

    temp=sorted(temp,key=lambda x: x[0],reverse=True)            
    return temp
 

def get_pair_blob_nonc(l,pixel_group,shape):
    indices=list()
    for s in range(0,len(pixel_group)-l):
        e=s+l-1
        if has_pair(s,e,pixel_group,shape):
            indices.append([s,l])
    return indices


def get_npair_full(blob,shape):
    res=list()
    for i in range(3,100,1):
        res1=get_npair_blob(i,blob,shape)
        for r in res1:
            res.append(r)
    res=sorted(res,key=lambda x: x[1],reverse=True)
    return res

def get_npair_blob(l,blob,shape):
    s=0
    e=step(s,blob,l-1)
    indices=list()
    step_val=1
    n=step_val
    while 1:
        np=has_npair(s,e,blob,shape)
        if np:
            indices.append([n-step_val,l])
        s=step(s,blob,step_val)
        e=step(e,blob,step_val)
        if n>=len(blob):
            break
        n+=step_val
    return indices


def find_sym_objects_nonc(blob,shape):
    found=list()
    ip_blobs=[blob]
    while len(ip_blobs)>0:
        ip_blob=ip_blobs.pop(-1)
        #print("len(ip_blob)=",len(ip_blob))

        if len(ip_blob)<4:
            print("found len=",len(ip_blob))
            found.append(ip_blob)
            continue
        res=get_symmetry_index_full_nonc(ip_blob,shape)
        s=res[0]
        e=s+res[1]-1
        if s>0 and e<(len(ip_blob)-1):
            print("found normal at",res)
            found.append(ip_blob[s:e+1])
            ip_blobs.append(ip_blob[:s])
            ip_blobs.append(ip_blob[e+1:])
        elif s==0 and e==(len(ip_blob)-1):
            print("found full at",res)
            found.append(ip_blob)
        elif s==0:
            print("found left at",res)
            found.append(ip_blob[:e+1])
            ip_blobs.append(ip_blob[e+1:])
        elif e==(len(ip_blob)-1):
            print("found right at",res)
            found.append(ip_blob[:s])
            ip_blobs.append(ip_blob[:s-1])

    # make objects continuous
    s=0
    l=0
    tot_len=0
    for f in found:
        tot_len+=len(f)
    print("tot_len=",tot_len)
    found_sorted=list()
    while s<tot_len:
        for f in found:
            if f[0]==blob[s]:
                found_sorted.append(f)
                break
        s+=len(found_sorted[-1])

    return found_sorted


def get_symmetry_index_full_nonc(pixel_group,shape):
    res=list()
    for i in range(3,len(pixel_group),1):
        res1=get_symmetry_index_blob_nonc(i,pixel_group,shape)
        for r in res1:
            res.append(r)


    return res

def get_best_index(index_list):
    #get sym objects with same index value order
    #by decreasing length
    index_list=sorted(index_list,key=lambda x: x[0])
    best_index=index_list[0][0]
    order_by_length=list()
    for r in index_list:
        if r[0]==best_index:
            order_by_length.append(r)
    order_by_length=sorted(order_by_length,key=lambda x: x[2],reverse=True)
    return order_by_length
    return [order_by_length[0][1],order_by_length[0][2]]


def get_symmetry_index_blob_nonc(l,pixel_group,shape):
    indices=list()
    if l>=len(pixel_group):
        index=get_symmetry_index(0,len(pixel_group)-1,pixel_group,shape)
        #index=(index/len(pixel_group))*100
        
        tot_dist=total_dist(0,len(pixel_group)-1,pixel_group,shape)+1
        index=(index/tot_dist)*100
        return [index,0,len(pixel_group)]
    for s in range(0,len(pixel_group)-l):
        e=s+l-1
        index=get_symmetry_index(s,e,pixel_group,shape)
        #index=(index/l)*100
        tot_dist=total_dist(s,e,pixel_group,shape)+1
        index=(index/tot_dist)*100
        indices.append([index,s,l])
    return indices


def get_symmetry_index_full(blob,shape):
    res=list()
    for i in range(3,len(blob),1):
        res1=get_symmetry_index_blob(i,blob,shape)
        for r in res1:
            res.append(r)
    res=sorted(res,key=lambda x: x[0])
    return res
    #order by decreasing length among best index
    best_index=res[0][0]
    order_by_length=list()
    for r in res:
        if r[0]==best_index:
            order_by_length.append(r)
    order_by_length=sorted(order_by_length,key=lambda x: x[2],reverse=True)
    return [order_by_length[0][1],order_by_length[0][2]]

def get_symmetry_index_blob(l,blob,shape):
    s=0
    e=step(s,blob,l-1)
    indices=list()
    step_val=1
    n=step_val
    while 1:
        index=get_symmetry_index(s,e,blob,shape)
        #index=(index/l)*100 #optional
        indices.append([index,n-step_val,l])
        s=step(s,blob,step_val)
        e=step(e,blob,step_val)
        if n>=len(blob):
            break
        n+=step_val
    return indices


def get_symmetry_index(s,e,blob,shape):
    res=-1
    l=next_i(s,blob)
    if l==e:
        return 0
    r=prev_i(e,blob)
    if l==r:
        l_dist=dist_i(blob[s],blob[l],shape)
        r_dist=dist_i(blob[r],blob[e],shape)
        diff=abs(l_dist-r_dist)
        return diff
    while 1:
        l_dist1=dist_i(blob[s],blob[l],shape)
        l_dist2=dist_i(blob[s],blob[r],shape)
        r_dist1=dist_i(blob[l],blob[e],shape)
        r_dist2=dist_i(blob[r],blob[e],shape)
        diff1=abs(l_dist1-r_dist2)
        diff2=abs(l_dist2-r_dist1)
        if diff1>res:
            res=diff1
        if diff2>res:
            res=diff2
        l=next_i(l,blob)
        if l==r:
            return res
        if next_i(l,blob)==r:
            l_dist=dist_i(blob[s],blob[l],shape)
            r_dist=dist_i(blob[l],blob[e],shape)
            diff=abs(l_dist-r_dist)
            if diff>res:
                return diff
            else:
                return res
        r=prev_i(r,blob)

def total_dist(s,e,blob,shape):
    dist=0
    pre=s
    while 1:
        cur=next_i(pre,blob)
        if cur==e:
            dist+=dist_i(blob[pre],blob[e],shape)
            break
        dist+=dist_i(blob[pre],blob[cur],shape)
        pre=cur
    return dist

def is_sym(start,end,pixels,threshold,shape):
    l=next_i(start,pixels)
    r=prev_i(end,pixels)
    if l==r:
        if abs((dist_i(pixels[start],pixels[l],shape))-\
                (dist_i(pixels[l],pixels[end],shape)))<threshold or per_dist_i(pixels[start],pixels[end],pixels[l],
                        shape)<threshold:
            return per_dist_i(pixels[start],pixels[end],pixels[l],shape)
        else: return -1
    while 1:
        if abs((dist_i(pixels[start],pixels[l],shape))-\
                (dist_i(pixels[r],pixels[end],shape)))>threshold and per_dist_i(pixels[start],pixels[end],pixels[l],
                        shape)>threshold:
                    return -1
        if abs((dist_i(pixels[l],pixels[end],shape))-\
                (dist_i(pixels[start],pixels[r],shape)))>threshold and per_dist_i(pixels[start],pixels[end],pixels[r],
                    shape)>threshold:
                    return -1
        l=next_i(l,pixels)
        if l==r:
            return per_dist_i(pixels[start],pixels[end],pixels[l],shape)
        if next_i(l,pixels)==r:
            if abs((dist_i(pixels[start],pixels[l],shape))-\
                (dist_i(pixels[l],pixels[end],shape)))<threshold or per_dist_i(pixels[start],pixels[end],pixels[l],
                        shape)<threshold:
                return per_dist_i(pixels[start],pixels[end],pixels[l],shape)
            else: return -1
        r=prev_i(r,pixels)
 

def carry_forward(lst):
    cf_array=list()
    cf=lst[-1]+lst[0]
    cf_array.append(cf)
    for i in range(1,len(lst)):
        cf=cf+lst[i]
        cf_array.append(cf)
    return cf_array

def count_zeros(lst):
    c=0
    for i in range(0,len(lst)):
        if lst[i]==0:
            c+=1
    return c

def count_nonzeros(lst):
    c=0
    for i in range(0,len(lst)):
        if lst[i]!=0:
            c+=1
    return c


def get_add_array(lst):
    add_array=list()
    a=lst[-1]+lst[0]
    add_array.append(a)
    for i in range(1,len(lst)):
        add_array.append(lst[i-1]+lst[i])
    return add_array


def remove_multi_zero(lst):
    no_multi_zero=list()
    no_multi_zero.append(lst[0])
    for i in range(1,len(lst)):
        if lst[i]==0 and no_multi_zero[-1]==0:
            pass
        else:
            no_multi_zero.append(lst[i])
    return no_multi_zero


def compare_array_deep(lst1,lst2):
    if len(lst1)!=len(lst2):
        return -1
    rotated=lst2
    max_diff=compare_array(lst1,rotated)
    min_diff=max_diff
    min_diff_array=lst2
    for i in range(1,len(lst1)):
        rotated=rotate_array_cw(rotated)
        max_diff=compare_array(lst1,rotated)
        if max_diff<min_diff:
            min_diff=max_diff
            min_diff_array=rotated
    print(min_diff)
    return min_diff_array

def compare_array(lst1,lst2):
    if len(lst1)!=len(lst2):
        return -1
    result=list(map(sub,lst2,lst1))
    return max(result)

def rotate_array_cw(lst):
    rotated=list()
    rotated.append(lst[-1])
    for i in range(0,len(lst)-1):
        rotated.append(lst[i])
    return rotated

def get_angle_diff(lst):
    diff_array=list()
    diff=angle_diff(lst[-1],lst[0])
    diff_array.append(diff)
    for i in range(1,len(lst)):
        diff=angle_diff(lst[i-1],lst[i])
        diff_array.append(diff)
    
    return diff_array

def get_angle_array(blob,shape):
    angle_array=list()
    a=angle_i(blob[-1],blob[0],shape)
    angle_array.append(a)
    for i in range(1,len(blob)):
        a=angle_i(blob[i-1],blob[i],shape)
        angle_array.append(a)
    return angle_array
 
def get_dist_array(blob,shape):
    dist_array=list()
    d=dist_i(blob[-1],blob[0],shape)
    dist_array.append(d)
    for i in range(1,len(blob)):
        d=dist_i(blob[i-1],blob[i],shape)
        dist_array.append(d)
    return dist_array


def get_unique_counter():
    global unique_counter
    unique_counter+=1
    return unique_counter

def is_increasing(s,e,blob,shape):
    # ab is line joining s and e
    # xy is line perpendicular to ab at point s
    # function checks if perp distance between xy and
    # all points between s and e is increasing or not.

    # find slope of line joining s,e
    p1=i_to_p(blob[s],shape)
    p2=i_to_p(blob[e],shape)
    slope=(float)(p2[0]-p1[0])/(float)(p2[1]-p1[1]+0.00000000000001)
    m=-1/(slope+0.00000000001)
    c=p1[0]-m*p1[1]

    pd_prev=0
    pd_list=list()
    i=next_i(s,blob)
    thresh=2
    pp=i_to_p(blob[i],shape)
    while 1:
        pd=(m*pp[1]-pp[0]+c)/sqrt(pow(m,2)+1)
        if (pd-pd_prev)>thresh:
            pd_list.append(1)
        elif (pd_prev-pd)>thresh:
            pd_list.append(-1)
        else:
            pd_list.append(0)
        pd_prev=pd
        i=next_i(i,blob)
        if i==e:
            break
        pp=i_to_p(blob[i],shape)
    #print(m,c)
    #print(p1,p2)
    print(pd_list)
    if pd_list.count(1)>0 and pd_list.count(-1)>0 or pd_list.count(0)>0:
        return 0
    else:
        return 1

def find_increasing_acw(s,blob,shape):
    n=5
    s_=step(s,blob,-5)
    while 1:
        if is_increasing(s_,s,blob,shape) and n<len(blob):
            n+=1
            s_=prev_i(s_,blob)
        else:
            break
    #print(i_between(next_i(s,blob),e,blob))
    return (next_i(s_,blob),s)

def find_increasing_cw(s,blob,shape):
    n=5
    e_=step(s,blob,5)
    while 1:
        if is_increasing(s,e_,blob,shape) and n<len(blob):
            n+=1
            e_=next_i(e_,blob)
        else:
            break
    #print(i_between(s,prev_i(e,blob),blob))
    return (s,prev_i(e_,blob))



def get_largest_line_quick(blob,shape):
    start=time.time()
    l1=len(blob)
    l3=3
    l2=int((l1+l3)/2)
    while 1:
        print(l1,l2,l3)
        if has_line_of_len(blob,l2,shape):
            if l1>l3:
                l3=l2
                l2=int((l1+l3)/2)
            else:
                l1=l2
                l2=int((l1+l3)/2)
        else:
            if l1<l3:
                l3=l2
                l2=int((l1+l3)/2)
            else:
                l1=l2
                l2=int((l1+l3)/2)
        if l1==l2 or l3==l2:
            break
    for s in range(0,len(blob)):
        e=step(s,blob,l2-1)
        if is_line(s,e,blob,shape):
            print("found quick line in",time.time()-start,"seconds. l=",l2)
            return (s,l2)

def is_line(s,e,blob,shape):
    max_pd=get_max_per_dist(s,e,blob,shape)
    if max_pd<(0.05*dist_i(blob[s],blob[e],shape)):
        return True
    return False

def has_line_of_len(blob,l,shape):
    for s in range(0,len(blob)):
        e=step(s,blob,l-1)
        if is_line(s,e,blob,shape):
            return True
    return False

def has_line_of_len_nonc(blob,l,shape):
    for s in range(0,len(blob)-l):
        e=step(s,blob,l-1)
        if is_line(s,e,blob,shape):
            return True
    return False

def get_largest_line_quick_nonc(blob,shape):
    start=time.time()
    l1=len(blob)
    l3=3
    l2=int((l1+l3)/2)
    while 1:
        print(l1,l2,l3)
        if has_line_of_len_nonc(blob,l2,shape):
            if l1>l3:
                l3=l2
                l2=int((l1+l3)/2)
            else:
                l1=l2
                l2=int((l1+l3)/2)
        else:
            if l1<l3:
                l3=l2
                l2=int((l1+l3)/2)
            else:
                l1=l2
                l2=int((l1+l3)/2)
        if l1==l2 or l3==l2:
            break
    for s in range(0,len(blob)-l2):
        e=step(s,blob,l2-1)
        if is_line(s,e,blob,shape):
            print("found quick line in",time.time()-start,"seconds. l=",l2)
            return (s,l2)



def get_max_per_dist(s,e,blob,shape):
    i=s 
    max_pd=0
    while 1:
        i=next_i(i,blob)
        if i==e:
            break
        pd=per_dist_i(blob[s],blob[e],blob[i],shape)
        if pd>max_pd:
            max_pd=pd
    return max_pd

    pairs=list()
    while 1:
        s,e=find_same_side_both(e,blob,shape)
        n+=1
        if (s,e) in pairs:
            break
        else:
            pairs.append((s,e))
    return pairs


def find_same_side_both(start,end,blob,shape):
    i_prev=0
    s=start
    e=end
    while 1:
        s,e=find_same_side_cw(s,e,blob,shape)
        s,e=find_same_side_acw(s,e,blob,shape)

        i=i_between(s,e,blob)
        if i==i_prev:
            break
        else:
            i_prev=i
    
    j_prev=0
    s_=start
    e_=end
    while 1:
        s_,e_=find_same_side_acw(s_,e_,blob,shape)
        s_,e_=find_same_side_cw(s_,e_,blob,shape)
        j=i_between(s_,e_,blob)
        if j==j_prev:
            break
        else:
            j_prev=j
    
    if i_prev>=j_prev:
        return (s,e)
    else:
        return (s_,e_)

def find_same_side_acw(s,e,blob,shape):
    n=0
    s_=s
    while 1:
        if on_same_side(s_,e,blob,shape) and n<len(blob):
            n+=1
            s_=prev_i(s_,blob)
        else:
            break
    #print(i_between(next_i(s,blob),e,blob))
    return (next_i(s_,blob),e)

def find_same_side_cw(s,e,blob,shape):
    n=0
    e_=e
    while 1:
        if on_same_side(s,e_,blob,shape) and n<len(blob):
            n+=1
            e_=next_i(e_,blob)
        else:
            break
    #print(i_between(s,prev_i(e,blob),blob))
    return (s,prev_i(e_,blob))

def get_outer_corner(blob,shape):
    outer=list()
    for i in range(1,len(blob)-1):
        a=i_to_p(blob[i-1],shape)
        b=i_to_p(blob[i+1],shape)
        c=i_to_p(blob[i],shape)
        vi=((a[1]-b[1])*(a[0]-c[0])-(a[0]-b[0])*(a[1]-c[1]))
        if vi>0:
            outer.append(blob[i])
    a=i_to_p(blob[-2],shape)
    b=i_to_p(blob[0],shape)
    c=i_to_p(blob[-1],shape)
    vi=((a[1]-b[1])*(a[0]-c[0])-(a[0]-b[0])*(a[1]-c[1]))
    if vi>0:
        outer.append(blob[-1])

    a=i_to_p(blob[-1],shape)
    b=i_to_p(blob[1],shape)
    c=i_to_p(blob[0],shape)
    vi=((a[1]-b[1])*(a[0]-c[0])-(a[0]-b[0])*(a[1]-c[1]))
    if vi>0:
        outer.append(blob[0])

    return outer

def get_inner_corner(blob,shape):
    inner=list()
    for i in range(1,len(blob)-1):
        a=i_to_p(blob[i-1],shape)
        b=i_to_p(blob[i+1],shape)
        c=i_to_p(blob[i],shape)
        vi=((a[1]-b[1])*(a[0]-c[0])-(a[0]-b[0])*(a[1]-c[1]))
        if vi<0:
            inner.append(blob[i])
    a=i_to_p(blob[-2],shape)
    b=i_to_p(blob[0],shape)
    c=i_to_p(blob[-1],shape)
    vi=((a[1]-b[1])*(a[0]-c[0])-(a[0]-b[0])*(a[1]-c[1]))
    if vi<0:
        inner.append(blob[-1])

    a=i_to_p(blob[-1],shape)
    b=i_to_p(blob[1],shape)
    c=i_to_p(blob[0],shape)
    vi=((a[1]-b[1])*(a[0]-c[0])-(a[0]-b[0])*(a[1]-c[1]))
    if vi<0:
        inner.append(blob[0])

    return inner


def on_same_side(s,e,blob,shape):
    a=i_to_p(blob[s],shape)
    b=i_to_p(blob[e],shape)
    i=s
    val=list()
    while 1:
        i=next_i(i,blob)
        if i==e:
            break
        c=i_to_p(blob[i],shape)
        vi=((a[1]-b[1])*(a[0]-c[0])-(a[0]-b[0])*(a[1]-c[1]))
        if vi>0:
            val.append(1)
        elif vi<0:
            val.append(-1)
        else:
            val.append(0)

    #return val
    if val.count(1)>0 and val.count(-1)>0:
        return 0
    else:
        return 1

def get_npair_old(s,e,blob,shape):
    np=list()
    i=s
    last=0
    while(1):
        i=next_i(i,blob)
        if i==e:
            np.append(-1)
            break
        pd=int(per_dist_i(blob[s],blob[e],blob[i],shape))
        if pd>last:
            np.append(1)
        elif pd<last:
            np.append(-1)
        else:
            np.append(0)
        last=pd

    for i in range(0,len(np)-1):
        if np[i]!=0 and np[i+1]==0:
            np[i+1]=np[i]
    for i in range(len(np)-2,-1,-1):
        if np[i+1]!=0 and np[i]==0:
            np[i]=np[i+1]
    first=list()
    first.append(blob[s])
    for i in range(1,len(np)):
        if np[i]==-1 and np[i-1]==1:
            first.append(blob[step(s,blob,i)])
    first.append(blob[e])
    #print("found",len(first)-2,"npair[s].")
    return len(first)-2

def get_corner_pixels(blob,shape):
    c=list()
    for i in range(1,len(blob)-1):
        p1=i_to_p(blob[i-1],shape)
        p2=i_to_p(blob[i+1],shape)
        if p1[0]!=p2[0] and p1[1]!=p2[1]:
            c.append(blob[i])
    return c

def break_blob_curve_c(blob,threshold,shape):
    s,l=get_largest_curve_quick(blob,threshold,shape)
    return strip_between_c(s,step(s,blob,l),blob)

def has_sym_of_len_curve(blob,l,threshold,shape):            
    for s in range(0,len(blob)):
        e=step(s,blob,l-1)
        sl=is_sym(s,e,blob,threshold,shape)
        if sl>threshold:
            return True
    return False


def get_largest_curve_quick(blob,threshold,shape):
    start=time.time()
    l1=len(blob)
    l3=3
    l2=int((l1+l3)/2)
    while 1:
        print(l1,l2,l3)
        if has_sym_of_len(blob,l2,threshold,shape):
            if l1>l3:
                l3=l2
                l2=int((l1+l3)/2)
            else:
                l1=l2
                l2=int((l1+l3)/2)
        else:
            if l1<l3:
                l3=l2
                l2=int((l1+l3)/2)
            else:
                l1=l2
                l2=int((l1+l3)/2)
        if l1==l2 or l3==l2:
            break
    for s in range(0,len(blob)):
        e=step(s,blob,l2-1)
        sl=is_sym(s,e,blob,threshold,shape)
        if sl>threshold:
            print("found quick sym in",time.time()-start,"seconds. l=",l2)
            return (s,l2)


def get_largest_sym_quick(blob,threshold_,shape):
    start=time.time()
    l1=len(blob)
    l3=3
    l2=int((l1+l3)/2)
    threshold=threshold_
    while 1:
        #threshold=int(0.01*l2)
        print(l1,l2,l3,threshold)
        if has_sym_of_len(blob,l2,threshold,shape):
            if l1>l3:
                l3=l2
                l2=int((l1+l3)/2)
            else:
                l1=l2
                l2=int((l1+l3)/2)
        else:
            if l1<l3:
                l3=l2
                l2=int((l1+l3)/2)
            else:
                l1=l2
                l2=int((l1+l3)/2)
        if l1==l2 or l3==l2:
            break
    for s in range(0,len(blob)):
        e=step(s,blob,l2-1)
        #threshold=int(0.05*l2)
        sl=is_sym(s,e,blob,threshold,shape)
        if sl>=0:
            print("found quick sym in",time.time()-start,"seconds. l=",l2)
            return (s,l2)

def get_largest_sym_quick_nonc(blob,threshold_,shape):
    start=time.time()
    threshold=5
    l1=len(blob)
    l3=3
    l2=int((l1+l3)/2)
    while 1:
        print(l1,l2,l3)
        if has_sym_of_len_nonc(blob,l2,threshold,shape):
            if l1>l3:
                l3=l2
                l2=int((l1+l3)/2)
            else:
                l1=l2
                l2=int((l1+l3)/2)
        else:
            if l1<l3:
                l3=l2
                l2=int((l1+l3)/2)
            else:
                l1=l2
                l2=int((l1+l3)/2)
        if l1==l2 or l3==l2:
            break
    for s in range(0,len(blob)-l2):
        e=step(s,blob,l2-1)
        sl=is_sym(s,e,blob,threshold,shape)
        if sl>=0:
            print("found quick sym in",time.time()-start,"seconds. l=",l2)
            return (s,l2)


def has_sym_of_len(blob,l,threshold,shape):            
    for s in range(0,len(blob)):
        e=step(s,blob,l-1)
        sl=is_sym(s,e,blob,threshold,shape)
        if sl>=0:
            return True
    return False

def has_sym_of_len_nonc(blob,l,threshold,shape):
    for s in range(0,len(blob)-l):
        e=step(s,blob,l-1)
        sl=is_sym(s,e,blob,threshold,shape)
        if sl>=0 and sl<threshold:
            return True
    return False

def break_blob_c(blob,threshold,shape):
    s,l=get_largest_sym_quick(blob,threshold,shape)
    return strip_between_c(s,step(s,blob,l),blob)

def break_blob_nonc(blob,threshold,shape):
    s,l=get_largest_sym_quick_nonc(blob,threshold,shape)
    return strip_between_nonc(s,step(s,blob,l),blob)


def break_blobs_c(blobs,threshold,shape):
    super_blobs=list()
    for blob in blobs:
        blobs_=break_blob(blob,threshold,shape)
        for blob_ in blobs_:
            super_blobs.append(blob_)
        blobs_=list()
    return super_blobs

def break_blobs_nonc(blobs,threshold,shape):
    super_blobs=list()
    for blob in blobs:
        blobs_=break_blob_nonc(blob,threshold,shape)
        for blob_ in blobs_:
            super_blobs.append(blob_)
        blobs_=list()
    return super_blobs

def nth_break_blob_nonc(n,blob,threshold,shape):
    c=0
    super_blobs=break_blob_nonc(blob,threshold,shape)
    while 1:
        if c==n:
            return super_blobs
        else:
            super_blobs=break_blobs_nonc(super_blobs,threshold,shape)
            c+=1

def strip_between_c(s,e,blob):
    ss=s
    first=list()
    while 1:
        first.append(blob[ss])
        ss=next_i(ss,blob)
        if ss==e:
            first.append(blob[ss])
            break
    second=list()
    ee=next_i(e,blob)
    while 1:
        second.append(blob[ee])
        ee=next_i(ee,blob)
        if ee==s:
            break
    #return second
    return [first,second]

def strip_between_nonc(s,e,blob):
    blobs=list()
    
    temp=list()
    for i in range(0,s):
        temp.append(blob[i])
    blobs.append(temp)
 
    temp=list()
    for i in range(s,e+1):
        temp.append(blob[i])
    blobs.append(temp)
    
    temp=list()
    for i in range(e,len(blob)):
        temp.append(blob[i])
    blobs.append(temp)
 
    return blobs


def is_sym(start,end,pixels,threshold,shape):
    l=next_i(start,pixels)
    r=prev_i(end,pixels)
    if l==r:
        if abs((dist_i(pixels[start],pixels[l],shape))-\
                (dist_i(pixels[l],pixels[end],shape)))<threshold or per_dist_i(pixels[start],pixels[end],pixels[l],
                        shape)<threshold:
            return per_dist_i(pixels[start],pixels[end],pixels[l],shape)
        else: return -1
    while 1:
        if abs((dist_i(pixels[start],pixels[l],shape))-\
                (dist_i(pixels[r],pixels[end],shape)))>threshold and per_dist_i(pixels[start],pixels[end],pixels[l],
                        shape)>threshold:
                    return -1
        if abs((dist_i(pixels[l],pixels[end],shape))-\
                (dist_i(pixels[start],pixels[r],shape)))>threshold and per_dist_i(pixels[start],pixels[end],pixels[r],
                    shape)>threshold:
                    return -1
        l=next_i(l,pixels)
        if l==r:
            return per_dist_i(pixels[start],pixels[end],pixels[l],shape)
        if next_i(l,pixels)==r:
            if abs((dist_i(pixels[start],pixels[l],shape))-\
                (dist_i(pixels[l],pixels[end],shape)))<threshold or per_dist_i(pixels[start],pixels[end],pixels[l],
                        shape)<threshold:
                return per_dist_i(pixels[start],pixels[end],pixels[l],shape)
            else: return -1
        r=prev_i(r,pixels)
 

def i_between(start,end,pixels):
    c=1
    s=start
    while 1:
        s=next_i(s,pixels)
        if(s==end):
            break
        c+=1
    return c

def per_dist(a,b,c):
    denominator=sqrt(pow(a[1]-b[1],2)+pow(a[0]-b[0],2))
    if denominator==0:
        return 0
    return ((a[1]-b[1])*(a[0]-c[0])-(a[0]-b[0])*(a[1]-c[1]))/denominator


@cuda.jit(device=True)
def per_dist_abs(a0,a1,b0,b1,c0,c1):
    return abs((a1-b1)*(a0-c0)-(a0-b0)*(a1-c1))/cmath.sqrt(pow(a1-b1,2)+pow(a0-b0,2))



def per_dist_i(i1,i2,i3,shape):
    p1=i_to_p(i1,shape)
    p2=i_to_p(i2,shape)
    p3=i_to_p(i3,shape)
    return per_dist(p1,p2,p3)

def find_angle_i(a,b,shape):
    p1=i_to_p(a,shape)
    p2=i_to_p(b,shape)
    return find_angle(p1,p2)

def find_angle(p1,p2):
    angle=math.atan2(p1[1]-p2[1],p1[0]-p2[0])*180/math.pi
    if angle<0:
        angle+=360
    return angle

def find_angle_diff(angle_pre,angle):
    angle_pre_q=0
    angle_q=0
    if angle<90:
        angle_q=1
    elif angle>=90 and angle<180:
        angle_q=2
    elif angle>=180 and angle<270:
        angle_q=3
    elif angle>=270 and angle<360:
        angle_q=4
    
    if angle_pre<90:
        angle_pre_q=1
    elif angle_pre>=90 and angle_pre<180:
        angle_pre_q=2
    elif angle_pre>=180 and angle_pre<270:
        angle_pre_q=3
    elif angle_pre>=270 and angle_pre<360:
        angle_pre_q=4
 
    if angle_pre_q==1:
        if angle_q==3 and (angle_pre+180)<angle:
            diff=360-angle+angle_pre
        elif angle_q==4:
            diff=360-angle+angle_pre
        else:
            diff=(angle-angle_pre)
    elif angle_pre_q==2:
        if angle_q==4 and (angle_pre+180)<angle:
            diff=360-angle+angle_pre
        else:
            diff=(angle-angle_pre)
    elif angle_pre_q==3:
        if angle_q==1 and (angle+180)<angle_pre:
            diff=360-angle_pre+angle
        else:
            diff=(angle-angle_pre)
    elif angle_pre_q==4:
        if angle_q==2 and (angle+180)<angle_pre:
            diff=360-angle_pre+angle
        elif angle_q==1:
            diff=360-angle_pre+angle
        else:
            diff=(angle-angle_pre)
    return diff

def angle_diff(a,b):
    diff=b-a
    if diff>180:
        diff=diff-360
    elif diff<-180:
        diff=diff+360
    return diff


def angle(a,b):
    """slope of line ab"""
    ang=math.degrees(math.atan2(a[1]-b[1],a[0]-b[0]))
    #return int(round(ang))
    return ang

def angle_i(i1,i2,shape):
    """slope of line ab"""
    a=i_to_p(i1,shape)
    b=i_to_p(i2,shape)
    return angle(a,b)

def get_slope(p1,p2):
    slope=(float)(p2[0]-p1[0])/(float)(p2[1]-p1[1]+0.00001)
    return slope

def get_slope_i(i1,i2,shape):
    p1=i_to_p(i1,shape)
    p2=i_to_p(i2,shape)
    return get_slope(p1,p2)

def len_between(p1,p2):
    if(p1[0]==p2[0]):
        return abs(p1[1]-p2[1])
    else: return abs(p1[0]-p2[0])

def len_between_i(i1,i2,shape):
    p1=i_to_p(i1,shape)
    p2=i_to_p(i2,shape)
    return len_between(p1,p2)

def dist(p1,p2):
    return sqrt(float(pow(p2[0]-p1[0],2)+pow(p2[1]-p1[1],2)))

def dist_i(i1,i2,shape):
    p1=i_to_p(i1,shape)
    p2=i_to_p(i2,shape)
    return dist(p1,p2)

def sort_blob_mix(blob_mix,shape):
    blank_img=new_image(shape)
    temp_blob_mix=blob_mix[:]
    draw_pixels_i(temp_blob_mix,15,blank_img)
    blobs=list()
    blobs.append(list())
    i=0
    count=0
    blobs[i].append(temp_blob_mix[-1])
    while 1:
        count+=1
        if(get_color_i(get_north_i(blobs[i][-1],shape[1]),blank_img)==15):
            draw_pixel_i(blobs[i][-1],200,blank_img)
            temp_blob_mix.remove(blobs[i][-1])
            blobs[i].append(get_north_i(blobs[i][-1],shape[1]))
        elif(get_color_i(get_east_i(blobs[i][-1],shape[1]),blank_img)==15):
            draw_pixel_i(blobs[i][-1],200,blank_img)
            temp_blob_mix.remove(blobs[i][-1])
            blobs[i].append(get_east_i(blobs[i][-1],shape[1]))
        elif(get_color_i(get_south_i(blobs[i][-1],shape[1]),blank_img)==15):
            draw_pixel_i(blobs[i][-1],200,blank_img)
            temp_blob_mix.remove(blobs[i][-1])
            blobs[i].append(get_south_i(blobs[i][-1],shape[1]))
        elif(get_color_i(get_west_i(blobs[i][-1],shape[1]),blank_img)==15):
            draw_pixel_i(blobs[i][-1],200,blank_img)
            temp_blob_mix.remove(blobs[i][-1])
            blobs[i].append(get_west_i(blobs[i][-1],shape[1]))
        else:
            draw_pixel_i(blobs[i][-1],200,blank_img)
            temp_blob_mix.remove(blobs[i][-1])
            blobs[i].reverse()
            i+=1
            if(count==len(blob_mix)):
                break
            blobs.append(list())
            blobs[i].append(temp_blob_mix[-1])
    print(len(blobs),"blob found.")
    return blobs
     
def condition_img(img):
    img_scaled=np.array(Image.new('L',(img.shape[1]*3,img.shape[0]*3),(255)))
    for r in range(0,img.shape[0]):
        for c in range(0,img.shape[1]):
            img_scaled[r*3][c*3]=img[r][c]
            img_scaled[r*3][c*3+1]=img[r][c]
            img_scaled[r*3][c*3+2]=img[r][c]
            img_scaled[r*3+1][c*3]=img[r][c]
            img_scaled[r*3+1][c*3+1]=img[r][c]
            img_scaled[r*3+1][c*3+2]=img[r][c]
            img_scaled[r*3+2][c*3]=img[r][c]
            img_scaled[r*3+2][c*3+1]=img[r][c]
            img_scaled[r*3+2][c*3+2]=img[r][c]
    return img_scaled


def get_color(p,img):
    return img[p[0]][p[1]]
def get_color_i(pi,img):
    r=int(pi/img.shape[1])
    c=pi%img.shape[1]
    return img[r][c]
def get_north(p):
    return [p[0]-1,p[1]]
def get_east(p):
    return [p[0],p[1]+1]
def get_south(p):
    return [p[0]+1,p[1]]
def get_west(p):
    return [p[0],p[1]-1]
def pixel_equals(p1,p2):
    if p1[0]==p2[0] and p1[1]==p2[1]:
        return 1
    return 0
def get_north_i(pi,width):
    r=int(pi/width)
    c=pi%width
    return (r-1)*width+c
def get_east_i(pi,width):
    r=int(pi/width)
    c=pi%width
    return r*width+c+1
def get_south_i(pi,width):
    r=int(pi/width)
    c=pi%width
    return (r+1)*width+c
def get_west_i(pi,width):
    r=int(pi/width)
    c=pi%width
    return r*width+c-1
def i_to_p(i,shape):
    r=int(i/shape[1])
    c=i%shape[1]
    return [r,c]

@cuda.jit(device=True)
def i_to_p(i,shape):
    r=int(i/shape[1])
    c=i%shape[1]
    return [r,c]

@cuda.jit(device=True)
def find_pd_abs(a,b,c,shape):
    a0=int(a/shape[1])
    a1=a%shape[1]
    b0=int(b/shape[1])
    b1=b%shape[1]
    c0=int(c/shape[1])
    c1=c%shape[1]
    return per_dist_abs(a0,a1,b0,b1,c0,c1)

def p_to_i(p,shape):
    return p[0]*shape[1]+p[1]

@cuda.jit(device=True)
def p_to_i(p,shape):
    return p[0]*shape[1]+p[1]


def angle_between(a,b,c):
    """angle between lines ab and bc"""
    ang=math.degrees(math.atan2(c[1]-b[1],c[0]-b[0])-math.atan2(a[1]-b[1],a[0]-b[0]))
    #return int(round(ang))
    return ang

def angle_between_i(ai,bi,ci,shape):
    a=i_to_p(ai,shape)
    b=i_to_p(bi,shape)
    c=i_to_p(ci,shape)
    return angle_between(a,b,c)

def next_i(i,pixels):
    if(i==len(pixels)-1):
        return 0
    else: return i+1

def prev_i(i,pixels):
    if i==0:
        return len(pixels)-1
    else: return i-1

def step(index,pixels,s):
    i=index
    if(s>0):
        ss=s
        while(ss>0):
            i=next_i(i,pixels)
            ss-=1
    else:
        ss=abs(s)
        while(ss>0):
            i=prev_i(i,pixels)
            ss-=1
    return i


def read_image(img):
    """ blob_dict={color: [[pixels],[pixels]]"""
    blob_dict={}
    for r in range(1,img.shape[0]-2):
        for c in range(1,img.shape[1]-2):
            if img[r][c]!=img[r][c+1]: # left ro right
                if int(img[r][c]) in blob_dict:
                    blob_dict[int(img[r][c])].add(r*img.shape[1]+c)
                else:
                    blob_dict[int(img[r][c])]=set()
                    blob_dict[int(img[r][c])].add(r*img.shape[1]+c)


                if int(img[r][c+1]) in blob_dict:
                    blob_dict[int(img[r][c+1])].add(r*img.shape[1]+c+1)
                else:
                    blob_dict[int(img[r][c+1])]=set()
                    blob_dict[int(img[r][c+1])].add(r*img.shape[1]+c+1)

            if img[r][c]!=img[r+1][c]: # top to bottom
                if int(img[r][c]) in blob_dict:
                    blob_dict[int(img[r][c])].add(r*img.shape[1]+c)
                else:
                    blob_dict[int(img[r][c])]=set()
                    blob_dict[int(img[r][c])].add(r*img.shape[1]+c)


                if int(img[r+1][c]) in blob_dict:
                    blob_dict[int(img[r+1][c])].add((r+1)*img.shape[1]+c)
                else:
                    blob_dict[int(img[r+1][c])]=set()
                    blob_dict[int(img[r+1][c])].add((r+1)*img.shape[1]+c)

            if img[r][c]!=img[r+1][c+1]: # diagonal
                if int(img[r][c]) in blob_dict:
                    blob_dict[int(img[r][c])].add(r*img.shape[1]+c)
                else:
                    blob_dict[int(img[r][c])]=set()
                    blob_dict[int(img[r][c])].add(r*img.shape[1]+c)


                if int(img[r+1][c+1]) in blob_dict:
                    blob_dict[int(img[r+1][c+1])].add((r+1)*img.shape[1]+c+1)
                else:
                    blob_dict[int(img[r+1][c+1])]=set()
                    blob_dict[int(img[r+1][c+1])].add((r+1)*img.shape[1]+c+1)

            if img[r+1][c]!=img[r][c+1]: # diagonal
                if int(img[r+1][c]) in blob_dict:
                    blob_dict[int(img[r+1][c])].add((r+1)*img.shape[1]+c)
                else:
                    blob_dict[int(img[r+1][c])]=set()
                    blob_dict[int(img[r+1][c])].add((r+1)*img.shape[1]+c)

                if int(img[r][c+1]) in blob_dict:
                    blob_dict[int(img[r][c+1])].add(r*img.shape[1]+c+1)
                else:
                    blob_dict[int(img[r][c+1])]=set()
                    blob_dict[int(img[r][c+1])].add(r*img.shape[1]+c+1)
    for c,blobs in blob_dict.items():
        blob_dict[c]=list(blobs)
    print("unsorted dict created.")
    return blob_dict


def blob_dict_summary(blob_dict):
    n=0
    for key,val in blob_dict.items():
        print (key,len(val))
        n+=len(val)
    print(len(blob_dict),"colors found.")
    print(n,"bounds found.")

def write_image(fname,image):
    Image.fromarray(image).save(fname)

def open_image(fname):
    """returns the image as numpy array"""
    return np.array(Image.open(fname).convert('L'))
def new_image(shape):
    return np.array(Image.new('L',(shape[1],shape[0]),(255)))

def get_pixels_from(s,e,blob):
    pixels=list()
    while 1:
        pixels.append(blob[s])
        s=next_i(s,blob)
        if s==e:
            pixels.append(blob[s])
            break
    return pixels

def draw_pixels_force(pixels,i,img):
    for p in pixels:
        draw_pixel_force(p,i,img)
def draw_pixels_i_force(pixels,i,img):
    for p in pixels:
        draw_pixel_i_force(p,i,img)

def draw_pixel_force(p,i,img):
    if get_color(p,img)==i:
        print("force draw pixel")
        img[p[0]][p[1]]+=20
    else:
        img[p[0]][p[1]]=i
def draw_pixel_i_force(p,i,img):
    r=int(p/img.shape[1])
    c=p%img.shape[1]
    draw_pixel_force((r,c),i,img)

def draw_pixel(p,i,img):
    if(get_color(p,img)==i):
        print("alert: re-draw")
    else: 
        img[p[0]][p[1]]=i
def draw_pixel_i(p,i,img):
    r=int(p/img.shape[1])
    c=p%img.shape[1]
    if(img[r][c]==i):
        print("alert: re-draw")
    else: img[r][c]=i

def draw_blobs_i(blobs,i1,i2,img):
    for even in range(0,len(blobs),2):
        draw_pixels_i(blobs[even],i1,img)
    for odd in range(1,len(blobs),2):
        draw_pixels_i(blobs[odd],i2,img)

def sort_dict_and_write_json(img,fname):
    start=time.time()
    blob_dict=read_image(img)
    for c,val in blob_dict.items():
        blob_dict[c]=sort_blob_mix(blob_dict[c],img.shape)
    write_json(blob_dict,fname+".sort_dict.txt")
    write_json(img.shape,fname+".shape.txt")
    print ("sorted in",time.time()-start,"s.")

def write_json(blob_dict,fname):
    json.dump(blob_dict,open(fname,'w'))
    print(fname,"written.")

def read_shape(fname):
    return json.load(open(fname+".shape.txt"))

def read_json(fname):
    """ note that the key is read as string."""
    raw_read=json.load(open(fname+".sort_dict.txt"))
    sort_dict={}
    for k,v in raw_read.items():
        sort_dict[int(k)]=v
    return sort_dict

def read_json_all(fname):
    """ note that the key is read as string."""
    raw_read=json.load(open(fname))
    return raw_read


def draw_blob_dict(blob_dict,shape,out_location):
    new_img=new_image(shape)
    for k,blobs in blob_dict.items():
        for blob in blobs:
            draw_pixels(blob,k+1,new_img)
    write_image(out_location,new_img)

def draw_pixels_len(pixels,s,l,c,img):
    ss=s
    ee=step(ss,pixels,l)
    draw_pixel_i(pixels[ss],c,img)
    while 1:
        ss=next_i(ss,pixels)
        draw_pixel_i(pixels[ss],c,img)
        if ss==ee:
            break

def draw_pixels(pixels,i,img):
    for p in pixels:
        draw_pixel(p,i,img)
def draw_pixels_i(pixels,i,img):
    for p in pixels:
        draw_pixel_i(p,i,img)


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

if __name__=="__main__":
    main()
