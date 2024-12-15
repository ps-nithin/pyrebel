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
import os.path
import os
from collections import Counter
from numba import float32,int32
import numba
os.environ['NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS']="0"
os.environ['NUMBA_ENABLE_CUDASIM']="0"
#from numba.core.errors import NumbaPerformanceWarning
#import warnings
#warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
from pdb import set_trace

parser=argparse.ArgumentParser()
parser.add_argument("-i","--input",help="Input file name. Defaults to camera stream.")
parser.add_argument("-t","--threshold",type=int,help="Threshold. Higher number gives higher resolution.")
parser.add_argument("-b","--blob",help="Selects the blob.")
parser.add_argument("-s","--symbol",help="Searches for the symbol.")
parser.add_argument("-o","--output",help="Output filename.")
parser.add_argument("-l","--layer",type=int,help="Selects the layer of bound abstraction.")
parser.add_argument("-n","--learn",help="Name of signature.")
parser.add_argument("-r","--recognize",help="Recognize the signature.")
parser.add_argument("-v","--vocabulary",help="Prints learned symbols.")
parser.add_argument("-c","--camera",help="Stream from camera.")
parser.add_argument("-p","--preprocess",help="Preprocess the input image.")
parser.add_argument("-i1","--index1",help="Index 1.")
parser.add_argument("-i2","--index2",help="Index 2.")
# create video sources & outputs
input = videoSource("csi://0", options={'width':320,'height':240,'framerate':30,'flipMethod':'rotate-270'})
output = videoOutput("", argv=sys.argv)
output2 = videoOutput("", argv=sys.argv)

if not os.path.exists('know_base.pkl'):
    fp=open('know_base.pkl','x')
    fp.close()
if not os.path.exists('vocabulary.pkl'):
    fp=open('vocabulary.pkl','x')
    fp.close()


import pickle
with open("know_base.pkl","rb") as fpr:
    try:
        know_base=pickle.load(fpr)
    except EOFError:
        know_base={}
with open("vocabulary.pkl","rb") as fpr:
    try:
        vocabulary=pickle.load(fpr)
    except EOFError:
        vocabulary=set()

abs_color=2

# process frames until EOS or the user exits
def main():
    args=parser.parse_args()
    if args.vocabulary:
        print(vocabulary)
        return
    if args.threshold:
        thresh=args.threshold
    else:
        thresh=5
    if args.blob:
        blob_index1=int(args.blob)
    else:
        blob_index1=1
    if args.symbol:
        search_symbol=args.symbol
    else:
        search_symbol=None
    
    if args.layer:
        layer_n=args.layer
    else:
        layer_n=0
    if args.recognize:
        recognize=1
        recognized=list()
        counter_recognized={}
    else:
        recognize=0
    if args.output:
        out_loc=args.output
    else:
        out_loc="output.png"
    if args.index1:
        index_1=int(args.index1)
    if args.index2:
        index_2=int(args.index2)


    if args.learn:
        learn_n=0
        if args.learn[-4:]=='.png':
            learn_single=True
            sign_name=args.learn
            ip_files_n=1
        else:
            learn_single=False
            ip_files=os.listdir(args.learn)
            ip_files_n=len(ip_files)


    frame_count=0
    while True:
        start_time=time.time()
        init_time=time.time()
        if args.learn:
            if learn_single:
                img_array=open_image(args.learn).astype('int32')
            else:
                sign_name=ip_files[learn_n]
                img_array=open_image(args.learn+ip_files[learn_n]).astype('int32')
        elif args.recognize:
            # checks if the argument recognize is a filename.
            if len(args.recognize)>1:
                img_array=open_image(args.recognize).astype('int32')
        elif args.preprocess:
            img_array=open_image(args.preprocess).astype('int32')

        if args.camera:
            # capture the next image
            img = input.Capture()
            if img is None: # timeout
                print("No camera capture!")
                continue  
            img_gray=convert_color(img,'gray8')
            img_array=cudaToNumpy(img_gray)
            cudaDeviceSynchronize()
            img_array=img_array.reshape(1,img_array.shape[0],img_array.shape[1])[0].astype('int32')
        
        """
        img_array=np.full([5,5],0,dtype=np.int)
        img_array[1][2]=5
        img_array[1][3]=5
        img_array[2][2]=5
        img_array[2][3]=5
        img_array[2][1]=5
        """
        img_array_d=cuda.to_device(img_array)
        img_array_orig=img_array
        img_array_orig_d=cuda.to_device(img_array_orig)
        shape_d=cuda.to_device(img_array.shape)
        ba_threshold=thresh
        threadsperblock=(16,16)
        blockspergrid_x=math.ceil(img_array.shape[0]/threadsperblock[0])
        blockspergrid_y=math.ceil(img_array.shape[1]/threadsperblock[1])
        blockspergrid=(blockspergrid_x,blockspergrid_y)
        
        img_wave_pre_init=np.zeros(img_array.shape[0]*img_array.shape[1],dtype=np.int32)
        img_wave_pre_init_d=cuda.to_device(img_wave_pre_init)
        preprocess_init[blockspergrid,threadsperblock](img_array_d,img_wave_pre_init_d)
        cuda.synchronize()
        bound_abstract_pre=np.zeros(img_array.shape[0]*img_array.shape[1],dtype=np.int32)
        bound_abstract_pre_d=cuda.to_device(bound_abstract_pre)
        initialize_ba_pre[img_array.shape[0]*img_array.shape[1],1](img_array_d,bound_abstract_pre_d)
        cuda.synchronize()
        
        bound_abstract_pre_h=bound_abstract_pre_d.copy_to_host()
        nz_ba_pre=get_non_zeros(bound_abstract_pre_h)
        nz_ba_pre_d=cuda.to_device(nz_ba_pre)

        ba_size_pre=np.full(img_array.shape[0],2,dtype=np.int32)
        ba_size_pre_d=cuda.to_device(ba_size_pre)
        ba_size_cum_pre_=np.cumsum(ba_size_pre)
        ba_size_cum_pre=np.delete(np.insert(ba_size_cum_pre_,0,0),-1)
        ba_size_cum_pre_d=cuda.to_device(ba_size_cum_pre)
        
        ba_max_pd_pre=np.zeros([len(nz_ba_pre),2],np.float64)
        ba_max_pd_pre_d=cuda.to_device(ba_max_pd_pre)
        ba_size_cum_pre_old=ba_size_cum_pre_[-1]
        pre_count=2
        while 1:
            find_ba_max_pd[len(nz_ba_pre),1](nz_ba_pre_d,ba_size_pre_d,img_wave_pre_init_d,ba_max_pd_pre_d,shape_d)
            cuda.synchronize()
            ba_max_pd_pre_h=ba_max_pd_pre_d.copy_to_host()
            find_next_ba_all[len(ba_size_pre),1](ba_max_pd_pre_d,ba_size_pre_d,ba_size_cum_pre_d,bound_abstract_pre_d,ba_threshold)
            cuda.synchronize()


            bound_abstract_pre_h=bound_abstract_pre_d.copy_to_host()
            nz_ba_pre=get_non_zeros(bound_abstract_pre_h)
            nz_ba_pre_d=cuda.to_device(nz_ba_pre)
        
            ba_max_pd_pre=np.zeros([len(nz_ba_pre),2],np.float64)
            ba_max_pd_pre_d=cuda.to_device(ba_max_pd_pre)

            ba_size_pre=ba_size_pre_d.copy_to_host()
            ba_size_cum_pre_=np.cumsum(ba_size_pre)
            ba_size_cum_pre=np.delete(np.insert(ba_size_cum_pre_,0,0),-1)
            ba_size_cum_pre_d=cuda.to_device(ba_size_cum_pre)
            pre_count+=1
            if ba_size_cum_pre_[-1]==ba_size_cum_pre_old:
                ba_change_pre=np.zeros([len(nz_ba_pre)],dtype=np.float64)
                ba_change_pre_d=cuda.to_device(ba_change_pre)
                ba_sign_pre=np.zeros([len(nz_ba_pre)],dtype=np.int32)
                ba_sign_pre_d=cuda.to_device(ba_sign_pre)
                find_change[len(nz_ba_pre),1](ba_size_pre_d,ba_size_cum_pre_d,nz_ba_pre_d,img_wave_pre_init_d,shape_d,ba_change_pre_d,ba_sign_pre_d)
                cuda.synchronize()
                ba_change_pre_h=ba_change_pre_d.copy_to_host()
                ba_sign_pre_h=ba_sign_pre_d.copy_to_host()
                print("count=",pre_count,ba_size_cum_pre_[-1])
                break
            else:
                ba_size_cum_pre_old=ba_size_cum_pre_[-1]
        
        #print(np.count_nonzero(ba_sign_pre_h==1),np.count_nonzero(ba_sign_pre_h==-1))
        white_count=np.count_nonzero(ba_sign_pre_h==1)
        black_count=np.count_nonzero(ba_sign_pre_h==-1)
        if white_count>black_count:
            invert=1
        else:
            invert=0
        nz_ba_pre_draw_hor_d=nz_ba_pre_d
        decrement_by_one[len(nz_ba_pre_d),1](nz_ba_pre_draw_hor_d)
        cuda.synchronize()

        out_image_hor=np.zeros(img_array.shape,dtype=np.int32)
        out_image_hor_d=cuda.to_device(out_image_hor)
        draw_pixels_cuda2(nz_ba_pre_draw_hor_d,ba_sign_pre_d,invert,255,out_image_hor_d)
        
        img_array_rot=np.rot90(img_array,k=1,axes=(0,1))
        img_array=np.ascontiguousarray(img_array_rot)
        img_array_d=cuda.to_device(img_array)
        shape_d=cuda.to_device(img_array.shape)
        threadsperblock=(16,16)
        blockspergrid_x=math.ceil(img_array.shape[0]/threadsperblock[0])
        blockspergrid_y=math.ceil(img_array.shape[1]/threadsperblock[1])
        blockspergrid=(blockspergrid_x,blockspergrid_y)
        
        img_wave_pre_init=np.zeros(img_array.shape[0]*img_array.shape[1],dtype=np.int32)
        img_wave_pre_init_d=cuda.to_device(img_wave_pre_init)
        preprocess_init[blockspergrid,threadsperblock](img_array_d,img_wave_pre_init_d)
        cuda.synchronize()
        
        bound_abstract_pre=np.zeros(img_array.shape[0]*img_array.shape[1],dtype=np.int32)
        bound_abstract_pre_d=cuda.to_device(bound_abstract_pre)
        initialize_ba_pre[img_array.shape[0]*img_array.shape[1],1](img_array_d,bound_abstract_pre_d)
        cuda.synchronize()
        
        bound_abstract_pre_h=bound_abstract_pre_d.copy_to_host()
        nz_ba_pre=get_non_zeros(bound_abstract_pre_h)
        nz_ba_pre_d=cuda.to_device(nz_ba_pre)

        ba_size_pre=np.full(img_array.shape[0],2,dtype=np.int32)
        ba_size_pre_d=cuda.to_device(ba_size_pre)
        ba_size_cum_pre_=np.cumsum(ba_size_pre)
        ba_size_cum_pre=np.delete(np.insert(ba_size_cum_pre_,0,0),-1)
        ba_size_cum_pre_d=cuda.to_device(ba_size_cum_pre)
        
        ba_max_pd_pre=np.zeros([len(nz_ba_pre),2],np.float64)
        ba_max_pd_pre_d=cuda.to_device(ba_max_pd_pre)
        ba_size_cum_pre_old=ba_size_cum_pre_[-1]
        pre_count=2
        while 1:
            
            find_ba_max_pd[len(nz_ba_pre),1](nz_ba_pre_d,ba_size_pre_d,img_wave_pre_init_d,ba_max_pd_pre_d,shape_d)
            cuda.synchronize()
            ba_max_pd_pre_h=ba_max_pd_pre_d.copy_to_host()
            find_next_ba_all[len(ba_size_pre),1](ba_max_pd_pre_d,ba_size_pre_d,ba_size_cum_pre_d,bound_abstract_pre_d,ba_threshold)
            cuda.synchronize()

            bound_abstract_pre_h=bound_abstract_pre_d.copy_to_host()
            nz_ba_pre=get_non_zeros(bound_abstract_pre_h)
            nz_ba_pre_d=cuda.to_device(nz_ba_pre)
        
            ba_max_pd_pre=np.zeros([len(nz_ba_pre),2],np.float64)
            ba_max_pd_pre_d=cuda.to_device(ba_max_pd_pre)

            ba_size_pre=ba_size_pre_d.copy_to_host()
            ba_size_cum_pre_=np.cumsum(ba_size_pre)
            ba_size_cum_pre=np.delete(np.insert(ba_size_cum_pre_,0,0),-1)
            ba_size_cum_pre_d=cuda.to_device(ba_size_cum_pre)
            
            pre_count+=1
            if ba_size_cum_pre_[-1]==ba_size_cum_pre_old:
                ba_change_pre=np.zeros([len(nz_ba_pre)],dtype=np.float64)
                ba_change_pre_d=cuda.to_device(ba_change_pre)
                ba_sign_pre=np.zeros([len(nz_ba_pre)],dtype=np.int32)
                ba_sign_pre_d=cuda.to_device(ba_sign_pre)
                find_change[len(nz_ba_pre),1](ba_size_pre_d,ba_size_cum_pre_d,nz_ba_pre_d,img_wave_pre_init_d,shape_d,ba_change_pre_d,ba_sign_pre_d)
                cuda.synchronize()
                ba_change_pre_h=ba_change_pre_d.copy_to_host()
                ba_sign_pre_h=ba_sign_pre_d.copy_to_host()
                print("count=",pre_count,ba_size_cum_pre_[-1])
                break
            else:
                ba_size_cum_pre_old=ba_size_cum_pre_[-1]




        #quant_img_array=np.zeros(img_array_d.shape,dtype=np.int32)
        #quant_img_array_d=cuda.to_device(quant_img_array)


        #ba_quantize[len(nz_ba_pre),1](img_array_d,nz_ba_pre_d,quant_img_array_d)
        #cuda.synchronize()

        #print(np.count_nonzero(ba_sign_pre_h==1),np.count_nonzero(ba_sign_pre_h==-1))
        white_count=np.count_nonzero(ba_sign_pre_h==1)
        black_count=np.count_nonzero(ba_sign_pre_h==-1)
        if white_count>black_count:
            invert=1
        else:
            invert=0
 
        nz_ba_pre_draw_ver_d=nz_ba_pre_d
        decrement_by_one[len(nz_ba_pre_d),1](nz_ba_pre_draw_ver_d)
        cuda.synchronize()

        out_image_ver=np.zeros(img_array.shape,dtype=np.int32)
        out_image_ver_d=cuda.to_device(out_image_ver)
        draw_pixels_cuda2(nz_ba_pre_draw_ver_d,ba_sign_pre_d,invert,255,out_image_ver_d)
        cuda.synchronize()
        out_image_ver_h=out_image_ver_d.copy_to_host()
        out_image_ver_rot=np.ascontiguousarray(np.rot90(out_image_ver_h,k=3,axes=(0,1)))
        out_image_ver_rot_d=cuda.to_device(out_image_ver_rot)

        threadsperblock=(16,16)
        blockspergrid_x=math.ceil(img_array_orig.shape[0]/threadsperblock[0])
        blockspergrid_y=math.ceil(img_array_orig.shape[1]/threadsperblock[1])
        blockspergrid=(blockspergrid_x,blockspergrid_y)
        clone_image[blockspergrid,threadsperblock](out_image_ver_rot_d,out_image_hor_d,255)
        cuda.synchronize()
        #clean_quant_img[blockspergrid,threadsperblock](out_image_hor_d)
        #cuda.synchronize()
        

        out_image2=np.zeros(img_array_orig.shape,dtype=np.int32)
        out_image2_d=cuda.to_device(out_image2)
        abs_count=np.zeros(1,dtype=np.int32)
        abs_count_d=cuda.to_device(abs_count)
        clone_image2[blockspergrid,threadsperblock](img_array_orig_d,out_image_hor_d,out_image2_d,not invert,abs_count_d)
        cuda.synchronize()
        print("edge count=",abs_count_d.copy_to_host()[0])

        ip_flood_img_h=img_array_orig_d.copy_to_host()
        out_flood_img_h=out_image2_d.copy_to_host()

        img_array=ip_flood_img_h
        img_array_d=cuda.to_device(img_array)
        threadsperblock=(16,16)
        blockspergrid_x=math.ceil(img_array.shape[0]/threadsperblock[0])
        blockspergrid_y=math.ceil(img_array.shape[1]/threadsperblock[1])
        blockspergrid=(blockspergrid_x,blockspergrid_y)
        img_fenced_d=img_array_d
        fence_image[blockspergrid,threadsperblock](img_array_d,img_fenced_d)
        cuda.synchronize()
        scaled_shape=np.array([img_array.shape[0]*3,img_array.shape[1]*3])
        scaled_shape_d=cuda.to_device(scaled_shape)
        img_scaled_d=cuda.device_array(scaled_shape,dtype=np.int32)
        scale_img_cuda[blockspergrid,threadsperblock](img_fenced_d,img_scaled_d)
        cuda.synchronize()
        img_scaled_h=img_scaled_d.copy_to_host()
        
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
        bound_len_low=50
        bound_len_high=20000
        threadsperblock=(16,16)
        blockspergrid_x=math.ceil(scaled_shape[0]/threadsperblock[0])
        blockspergrid_y=math.ceil(scaled_shape[1]/threadsperblock[1])
        blockspergrid=(blockspergrid_x,blockspergrid_y)
        get_bound_cuda2[blockspergrid,threadsperblock](img_boundary_d,bound_len_low,bound_len_high,seed_map_d,bound_info_d)
        cuda.synchronize()
        
        binfo=bound_info_d.copy_to_host()
        a=binfo.transpose()[0]
        s=binfo.transpose()[1]
        nz_a=get_non_zeros(a)
        nz_s=get_non_zeros(s)

        
        print("len(nz_s)=",len(nz_s))
        #nz=np.column_stack((nz_a,nz_s))
        #nz_sort=nz[nz[:,1].argsort()]
        nz_s_cum_=np.cumsum(nz_s)
        nz_s_cum=np.delete(np.insert(nz_s_cum_,0,0),-1)
        nz_s_cum_d=cuda.to_device(nz_s_cum)
        nz_a_d=cuda.to_device(nz_a)
        nz_s_d=cuda.to_device(nz_s)

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
        
        nz_si_d=cuda.to_device(nz_s)
        increment_by_one[len(nz_s),1](nz_si_d)
        nz_si=nz_si_d.copy_to_host()
        nz_si_cum_=np.cumsum(nz_si)
        nz_si_cum=np.delete(np.insert(nz_si_cum_,0,0),-1)
        nz_si_cum_d=cuda.to_device(nz_si_cum)

        bound_data_d=cuda.device_array([nz_s_cum_[-1]],dtype=np.int32)
        get_bound_data_init[math.ceil(len(nz_a)/256),256](nz_a_d,nz_s_cum_d,img_boundary_d,bound_data_d)
        cuda.synchronize()

        dist_data_d=cuda.device_array([nz_s_cum_[-1]],dtype=np.float64)
        get_dist_data_init[math.ceil(nz_s_cum_[-1]/256),256](bound_data_d,img_boundary_d,dist_data_d)
        cuda.synchronize()
        
        max_dist_d=cuda.device_array([len(nz_s),2],dtype=np.int32)
        get_max_dist[math.ceil(len(nz_s)/1),1](nz_s_cum_d,nz_s_d,bound_data_d,dist_data_d,max_dist_d)
        cuda.synchronize()

        bound_data_ordered_d=cuda.device_array([nz_si_cum_[-1]],dtype=np.int32)
        bound_abstract=np.zeros([nz_si_cum_[-1]],dtype=np.int32)
        bound_abstract_d=cuda.to_device(bound_abstract)
        bound_threshold=np.zeros([nz_si_cum_[-1]],dtype=np.float64)
        bound_mark_d=cuda.device_array([nz_si_cum_[-1]],dtype=np.int32)
        bound_threshold_d=cuda.to_device(bound_threshold)
        ba_size=np.zeros([nz_si_cum_[-1]],dtype=np.int32)
        ba_size_d=cuda.to_device(ba_size)
        get_bound_data_order[math.ceil(len(nz_a)/256),256](max_dist_d,nz_si_cum_d,img_boundary_d,bound_abstract_d,bound_data_ordered_d,bound_threshold_d,bound_mark_d,ba_size_d,thresh)
        cuda.synchronize()

        bound_threshold_h=bound_threshold_d.copy_to_host()
        bound_abstract_h=bound_abstract_d.copy_to_host()
        nz_ba_h=get_non_zeros(bound_abstract_h)
        nz_ba_d=cuda.to_device(nz_ba_h)
        nz_ba_pre_size=len(nz_ba_h)
        ba_size_h=ba_size_d.copy_to_host()
        nz_ba_size_h=get_non_zeros(ba_size_h)
        nz_ba_size_d=cuda.to_device(nz_ba_size_h)
        nz_ba_size_cum_=np.cumsum(nz_ba_size_h)
        nz_ba_size_cum=np.delete(np.insert(nz_ba_size_cum_,0,0),-1)
        nz_ba_size_cum_d=cuda.to_device(nz_ba_size_cum)
        
        ba_change=np.zeros([len(nz_ba_size_h)],dtype=np.float64)
        ba_change_d=cuda.to_device(ba_change)

        ba_max_pd=np.zeros([len(nz_ba_h),2],np.float64)
        ba_max_pd_d=cuda.to_device(ba_max_pd)
        out_image=np.zeros(scaled_shape,dtype=np.int32)
        out_image_d=cuda.to_device(out_image)
        n=0
        bound_data_ordered_h=bound_data_ordered_d.copy_to_host()
        draw_pixels_cuda(bound_data_ordered_d,100,out_image_d)
        if args.learn:
            if len(nz_a)==3:
                print("Learning: true")
                print("Sign name:",sign_name)
                learn=1
            else:
                print("Invalid input")
                return
        else:
            learn=0
            print("Learning: false")
        
        if args.recognize:
            print("Recognize: true")
            recognize=1
            recognize_init=1
            for i in range(len(nz_a)):
                recognized.append(list())
        else:
            recognize=0
            print("Recognize: false")

        #print("init layer",nz_ba_size_h)
        sign_layer_union=[]
        inv_sign=1
        while 1:
            find_ba_max_pd[len(nz_ba_h),1](nz_ba_d,nz_ba_size_d,bound_data_ordered_d,ba_max_pd_d,scaled_shape_d)
            cuda.synchronize()
            ba_max_pd_h=ba_max_pd_d.copy_to_host()

            find_next_ba[len(nz_s),1](ba_max_pd_d,nz_ba_size_d,nz_ba_size_cum_d,bound_abstract_d,ba_threshold)
            cuda.synchronize()
            bound_abstract_h=bound_abstract_d.copy_to_host()
            nz_ba_h=get_non_zeros(bound_abstract_h)
            nz_ba_d=cuda.to_device(nz_ba_h)

            ba_max_pd=np.zeros([len(nz_ba_h),2],np.float64)
            ba_max_pd_d=cuda.to_device(ba_max_pd)

            nz_ba_size_h=nz_ba_size_d.copy_to_host()
            nz_ba_size_cum_=np.cumsum(nz_ba_size_h)
            nz_ba_size_cum=np.delete(np.insert(nz_ba_size_cum_,0,0),-1)
            nz_ba_size_cum_d=cuda.to_device(nz_ba_size_cum)

            ba_change=np.zeros([len(nz_ba_h)],dtype=np.float64)
            ba_change_d=cuda.to_device(ba_change)
            ba_sign=np.zeros([len(nz_ba_h)],dtype=np.int32)
            ba_sign_d=cuda.to_device(ba_sign)
            find_change[len(nz_ba_h),1](nz_ba_size_d,nz_ba_size_cum_d,nz_ba_d,bound_data_ordered_d,scaled_shape_d,ba_change_d,ba_sign_d)
            cuda.synchronize()
            ba_change_h=ba_change_d.copy_to_host()
            ba_sign_h=ba_sign_d.copy_to_host()
            #select_ba_change=ba_change_h[nz_ba_size_cum[blob_index]:nz_ba_size_cum[blob_index]+nz_ba_size_h[blob_index]]
            """
            for blob_i in range(len(nz_a)):
                cur_sign_list=set()
                select_ba_sign=ba_sign_h[nz_ba_size_cum[blob_i]:nz_ba_size_cum[blob_i]+nz_ba_size_h[blob_i]]
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
                        cur_sign_list.add(sign)
                        sign_inverse=inverse_sign(sign)
                        cur_sign_list.add(sign_inverse)
            #sign_name_layer=nz_ba_size_h[blob_index]
            #print("layer",n,":",nz_ba_size_h)
                if learn:
                    for cur_sign in cur_sign_list:
                        if cur_sign in know_base:
                            if sign_name in know_base[cur_sign]:
                                know_base[cur_sign][sign_name]+=1
                            else:
                                know_base[cur_sign][sign_name]=1
                        else:
                            know_base[cur_sign]={sign_name:1}

                if recognize:
                    for cur_sign in cur_sign_list:
                        if cur_sign in know_base:
                            symbol_recognized=list(know_base[cur_sign].keys())
                            recognized[blob_i]=recognized[blob_i]+symbol_recognized
            """

            if n==layer_n:
                #select_ba=nz_ba_h[nz_ba_size_cum[blob_index]:nz_ba_size_cum[blob_index]+nz_ba_size_h[blob_index]]
                #select_ba_d=cuda.to_device(select_ba)
                #temp_nz_ba_d=nz_ba_d
                #decrement_by_one[len(select_ba),1](select_ba_d)
                #draw_pixels_from_indices_cuda(select_ba_d,bound_data_ordered_d,255,out_image_d)

                #select_ba_change=ba_change_h[nz_ba_size_cum[blob_index]:nz_ba_size_cum[blob_index]+nz_ba_size_h[blob_index]]
                #print("layer=",n,"len(signature)=",len(select_ba_change)-1,"\n",select_ba_change[:-1])
                break
            n+=1
        
        if recognize:
            for blob_i in range(len(nz_a)):
                if len(recognized[blob_i])>0:
                    counter_recognized[Counter(recognized[blob_i]).most_common()[0][0]]=blob_i
            #print(counter_recognized)
            
            if search_symbol in counter_recognized:
                blob_index=counter_recognized[search_symbol]
                select_bound=bound_data_ordered_h[nz_si_cum[blob_index]:nz_si_cum[blob_index]+nz_si[blob_index]]
                select_bound_d=cuda.to_device(select_bound)
                draw_pixels_cuda(select_bound_d,200,out_image_d)
            else:
                print("Symbol not found.")
            recognized=list()
            counter_recognized={}

        fpr.close()
        with open('know_base.pkl','wb') as fpw:
            pickle.dump(know_base,fpw)
        with open('vocabulary.pkl','wb') as fpw:
            pickle.dump(vocabulary,fpw)

        print("len(know_base)=",len(know_base))
        out_image_h=out_image_d.copy_to_host()
        img_boundary_cuda=cudaFromNumpy(ip_flood_img_h)
        img_boundary_cuda_rgb=convert_color(img_boundary_cuda,'rgb8')
        
        img_boundary_cuda2=cudaFromNumpy(out_flood_img_h)
        img_boundary_cuda_rgb2=convert_color(img_boundary_cuda2,'rgb8')

        output_png=cudaToNumpy(img_boundary_cuda_rgb2)
        cuda.synchronize()
        write_image(out_loc,output_png)
        # render the image
        output.Render(img_boundary_cuda_rgb)
        output2.Render(img_boundary_cuda_rgb2)
        # exit on input/output EOS
        #if not input.IsStreaming() or not output.IsStreaming():
        #    break
        
        print("Finished in total of",time.time()-init_time,"seconds at",float(1/(time.time()-init_time)),"fps count=",frame_count)
        frame_count+=1
        """
        if args.learn:
            learn_n+=1
            if learn_n==ip_files_n:
                break
        else:
            break
        """


@cuda.jit
def clone_image(img_array,img_clone,color):
    r,c=cuda.grid(2)
    if r<img_array.shape[0] and c<img_array.shape[1]:
        if img_array[r][c]==color:
            img_clone[r][c]=color

@cuda.jit
def clone_image2(img_array_orig,image_to_clone,img_cloned,inv,count):
    r,c=cuda.grid(2)
    if r>0 and r<img_array_orig.shape[0] and c>0 and c<img_array_orig.shape[1]:
        if image_to_clone[r][c]==255:
            if inv:
                img_cloned[r][c]=img_array_orig[r][c]
            else:
                img_cloned[r][c]=255-img_array_orig[r][c]
            cuda.atomic.add(count,0,1)


@cuda.jit
def clean_quant_img(quant_img_d):
    r,c=cuda.grid(2)
    if r>0 and r<quant_img_d.shape[0]-1 and c>0 and c<quant_img_d.shape[1]-1:
        #if quant_img_d[r][c]!=0 and quant_img_d[r-1][c-1]==0 and quant_img_d[r-1][c]==0 and quant_img_d[r-1][c+1]==0 and quant_img_d[r][c-1]==0 and quant_img_d[r][c+1]==0 and quant_img_d[r+1][c-1]==0 and quant_img_d[r+1][c]==0 and quant_img_d[r+1][c+1]==0:
            #quant_img_d[r][c]=0
        if quant_img_d[r][c]==0:
            if quant_img_d[r-1][c]!=0 and quant_img_d[r][c+1]!=0:
                quant_img_d[r][c]=255
            elif quant_img_d[r][c+1]!=0 and quant_img_d[r+1][c]!=0:
                quant_img_d[r][c]=255
            elif quant_img_d[r+1][c]!=0 and quant_img_d[r][c-1]!=0:
                quant_img_d[r][c]=255
            elif quant_img_d[r][c-1]!=0 and quant_img_d[r-1][c]!=0:
                quant_img_d[r][c]=255


@cuda.jit
def ba_quantize(img_array_d,nz_ba_pre_d,quant_img_array_d):
    ci=cuda.grid(1)
    if ci<len(nz_ba_pre_d)-1:
        i1=nz_ba_pre_d[ci]-1
        i2=nz_ba_pre_d[ci+1]-1
        r=int(i1/img_array_d.shape[1])
        c1=i1%img_array_d.shape[1]
        c2=i2%img_array_d.shape[1]
        """
        for c in range(c1,c1+width):
            quant_img_array_d[r][c]=img_array_d[r][c1]
        for c in range(c1+width,c2+1):
            quant_img_array_d[r][c]=img_array_d[r][c2]
        """
        for c in range(c1,c2+1):
            quant_img_array_d[r][c]=int((img_array_d[r][c1]+img_array_d[r][c2])/2)

@cuda.jit
def initialize_ba_pre(img_array_d,bound_abstract_pre_d):
    ci=cuda.grid(1)
    if ci==0:
        bound_abstract_pre_d[ci]=ci+1
    elif ci<len(bound_abstract_pre_d) and ci%img_array_d.shape[1]==0:
        bound_abstract_pre_d[ci]=ci+1
        bound_abstract_pre_d[ci-1]=ci
    elif ci==len(bound_abstract_pre_d)-1:
        bound_abstract_pre_d[ci]=ci+1

@cuda.jit
def preprocess_init(img_array_d,img_wave_pre_init_d):
    r,c=cuda.grid(2)
    if r<img_array_d.shape[0] and c<img_array_d.shape[1]:
        img_wave_pre_init_d[r*img_array_d.shape[1]+c]=img_array_d[r][c]*img_array_d.shape[1]+c

@cuda.jit
def get_init_intensity_diff(img_array_d,r_intensity_diff_array_d,c_intensity_diff_array_d):
    r,c=cuda.grid(2)
    max_diff_r=0
    max_diff_c=0
    max_diff_ri=0
    max_diff_ci=0
    if r<img_array_d.shape[0] and c<img_array_d.shape[1]:
        for rr in range(0,img_array_d.shape[0]):
            cur_diff=abs(img_array_d[r][c]-img_array_d[rr][c])
            if cur_diff>max_diff_r:
                max_diff_r=cur_diff
                max_diff_ri==rr
            elif cur_diff==max_diff_r:
                if abs(r-rr)<abs(r-max_diff_ri):
                    max_diff_ri=rr

        c_intensity_diff_array_d[r][c][0]=max_diff_r
        c_intensity_diff_array_d[r][c][1]=max_diff_ri
        for cc in range(0,img_array_d.shape[1]):
            cur_diff=abs(img_array_d[r][c]-img_array_d[r][cc])
            if cur_diff>max_diff_c:
                max_diff_c=cur_diff
                max_diff_ci==cc
            elif cur_diff==max_diff_c:
                if abs(c-cc)<abs(c-max_diff_ci):
                    max_diff_ci=cc
        r_intensity_diff_array_d[r][c][0]=max_diff_c
        r_intensity_diff_array_d[r][c][1]=max_diff_ci
        
@cuda.jit
def get_max_intensity_diff(img_array_d,r_intensity_diff_array_d,c_intensity_diff_array_d,r_img_intensity_diff_d,c_img_intensity_diff_d):
    r,c=cuda.grid(2)
    r_max=0
    c_max=0
    c_max_ri1=0
    c_max_ri2=0
    r_max_ci1=0
    r_max_ci2=0
    if r<img_array_d.shape[0] and c<img_array_d.shape[1]:
        for rr in range(0,img_array_d.shape[0]):
            if c_intensity_diff_array_d[rr][c][0]>c_max:
                c_max=c_intensity_diff_array_d[rr][c][0]
                c_max_ri1=c_intensity_diff_array_d[rr][c][1]
                c_max_ri2=rr
        for cc in range(0,img_array_d.shape[1]):
            if r_intensity_diff_array_d[r][cc][0]>r_max:
                r_max=r_intensity_diff_array_d[r][cc][0]
                r_max_ci1=r_intensity_diff_array_d[r][cc][1]
                r_max_ci2=cc
        r_img_intensity_diff_d[r][r_max_ci2]=255
        c_img_intensity_diff_d[c_max_ri2][c]=255


def get_network_from_neighbor_data(nz_a,neighbor_data_h):
    network={}
    for r in range(0,len(nz_a)):
        network[nz_a[r]]=set(neighbor_data_h[r])
        network[nz_a[r]].remove(0)
    return network

def get_neighbors(graph,node):
    return graph.get(node,[])

def find_all_neighbors(graph,node,visited=None):
    visited=set()

    return visited

def find_color_range(cc_array):
    range_tuple=[0,0]
    for i in range(256):
        if cc_array[i]>50:
            range_tuple[0]=i
            break
    for i in range(255,-1,-1):
        if cc_array[i]>50:
            range_tuple[1]=i
            break
    return range_tuple

@cuda.jit
def get_neighbor_data_init2(tmp_img,binfo_map_d,seed_map_d,neighbor_dist,neighbor_data_d):
    # method to find neighbors
    rr,cc=cuda.grid(2)
    if rr>neighbor_dist and rr<tmp_img.shape[0]-neighbor_dist and cc>neighbor_dist and cc<tmp_img.shape[1]-neighbor_dist:
        color=tmp_img[rr][cc]
        n=0
        # condition for neighborhoodness
        for r in range(rr-neighbor_dist,rr+neighbor_dist):
            for c in range(cc-neighbor_dist,cc+neighbor_dist):
                if tmp_img[r][c]!=500 and tmp_img[r][c]==color and ((rr-r)**2+(cc-c)**2)<neighbor_dist**2:
                    neighbor_data_d[binfo_map_d[seed_map_d[r*tmp_img.shape[1]+c]]][n]=seed_map_d[rr*tmp_img.shape[1]+cc]
                    n+=1
        # end

@cuda.jit
def flood_fill4(img_boundary,color,threshold,out_flood_img):
    r,c=cuda.grid(2)
    if r<img_boundary.shape[0]-threshold and r>threshold and c<img_boundary.shape[1]-threshold and c>threshold and img_boundary[r][c]==color:
        for y in range(r-threshold,r+threshold):
            for x in range(c-threshold,c+threshold):
                d_cur=sqrt(float(pow(r-y,2)+pow(c-x,2)))
                if img_boundary[y][x]==color and d_cur<threshold and d_cur>5:
                    npoints=math.floor(d_cur)
                    m=(y-r)/(x-c)
                    cc=r-m*c # y=mx+cc -> cc=y-mx
                    step=(x-c)/(npoints-1)
                    if x==c:
                        for i in range(npoints):
                            xx=c
                            yy=round(r+i*step)
                            out_flood_img[yy][xx]=color
                    else:
                        for i in range(npoints):
                            xx=round(c+i*step)
                            yy=round(m*xx+cc)
                            out_flood_img[yy][xx]=color
  
  
@cuda.jit
def flood_fill5(img_boundary,color,threshold,out_flood_img):
    r,c=cuda.grid(2)
    if r<img_boundary.shape[0]-threshold and r>threshold and c<img_boundary.shape[1]-threshold and c>threshold and img_boundary[r][c]==color:
        rr=r
        cc=c
        for y in range(r-threshold,r+threshold):
            for x in range(c-threshold,c+threshold):
                d_cur=sqrt(float(pow(r-y,2)+pow(c-x,2)))
                if img_boundary[y][x]==color and d_cur<threshold:
                    dx=abs(x-cc)
                    dy=abs(y-rr)
                    sx=1 if cc<x else -1
                    sy=1 if rr<y else -1
                    err=dx-dy
                    while True:
                        out_flood_img[rr][cc]=color
                        if cc==x and rr==y:
                            break
                        e2=2*err
                        if e2>-dy:
                            err-=dy
                            cc+=sx
                        if e2<dx:
                            err+=dx
                            rr+=sy
                                                

@cuda.jit
def flood_fill3(img_boundary,color,threshold,out_flood_img):
    r,c=cuda.grid(2)
    if r<out_flood_img.shape[0]-threshold and r>threshold and c<out_flood_img.shape[1]-threshold and c>threshold and img_boundary[r][c]==color:
        for y in range(r-threshold,r+threshold):
            for x in range(c-threshold,c+threshold):
                d_cur=sqrt(float(pow(r-y,2)+pow(c-x,2)))
                if d_cur<threshold:
                    out_flood_img[y][x]=255


@cuda.jit
def flood_fill2(img_boundary,color,threshold,out_flood_img):
    r,c=cuda.grid(2)
    nbr_count=0
    init_run=True
    nbr_data=cuda.local.array(shape=(625,),dtype=numba.int32)
    if r<out_flood_img.shape[0]-threshold and r>threshold and c<out_flood_img.shape[1]-threshold and c>threshold and img_boundary[r][c]==color:
        for y in range(r-threshold,r+threshold):
            for x in range(c-threshold,c+threshold):
                d_cur=sqrt(float(pow(r-y,2)+pow(c-x,2)))
                if img_boundary[y][x]==color and d_cur<threshold:
                    if init_run:
                        init_run=False
                    else:
                        for nbri in nbr_data:
                            if nbri==0:
                                break
                            ynbr=int(nbri/out_flood_img.shape[1])
                            xnbr=nbri%out_flood_img.shape[1]
                            if r>=y and r>=ynbr:
                                ymax=r
                            elif y>=r and y>=ynbr:
                                ymax=y
                            else:
                                ymax=ynbr

                            if r<=y and r<=ynbr:
                                ymin=r
                            elif y<=r and y<=ynbr:
                                ymin=y
                            else:
                                ymin=ynbr

                            if c>=x and c>=xnbr:
                                xmax=c
                            elif x>=c and x>=xnbr:
                                xmax=x
                            else:
                                xmax=xnbr
                            
                            if c<=x and c<=xnbr:
                                xmin=c
                            elif x<=c and x<=xnbr:
                                xmin=x
                            else:
                                xmin=xnbr
                            x1,y1=c,r
                            x2,y2=x,y
                            x3,y3=xnbr,ynbr
                            for py in range(ymin,ymax):
                                for px in range(xmin,xmax):
                                    denominator=(y2-y3)*(x1-x3)+(x3-x2)*(y1-y3)
                                    alpha=((y2-y3)*(px-x3)+(x3-x2)*(py-y3))/denominator
                                    beta=((y3-y1)*(px-x3)+(x1-x3)*(py-y3))/denominator
                                    gamma=1-alpha-beta
                                    if 0<=alpha<=1 and 0<=beta<=1 and 0<=gamma<=1:
                                        out_flood_img[py][px]=255
                    nbr_data[nbr_count]=y*out_flood_img.shape[1]+x
                    nbr_count+=1



@cuda.jit
def flood_fill(ip_flood_img,color,threshold,out_flood_img):
    r,c=cuda.grid(2)
    nbr_count=0
    init_run=True
    nbr_data=cuda.local.array(shape=(625,),dtype=numba.int32)
    if r<ip_flood_img.shape[0]-threshold and r>threshold and c<ip_flood_img.shape[1]-threshold and c>threshold and ip_flood_img[r][c]==color:
        for y in range(r-threshold,r+threshold):
            for x in range(c-threshold,c+threshold):
                d_cur=sqrt(float(pow(r-y,2)+pow(c-x,2)))
                if ip_flood_img[y][x]==color and d_cur<threshold:
                    if init_run:
                        init_run=False
                    else:
                        for nbri in nbr_data:
                            if nbri==0:
                                break
                            ynbr=int(nbri/ip_flood_img.shape[1])
                            xnbr=nbri%ip_flood_img.shape[1]
                            if r>=y and r>=ynbr:
                                ymax=r
                            elif y>=r and y>=ynbr:
                                ymax=y
                            else:
                                ymax=ynbr

                            if r<=y and r<=ynbr:
                                ymin=r
                            elif y<=r and y<=ynbr:
                                ymin=y
                            else:
                                ymin=ynbr

                            if c>=x and c>=xnbr:
                                xmax=c
                            elif x>=c and x>=xnbr:
                                xmax=x
                            else:
                                xmax=xnbr
                            
                            if c<=x and c<=xnbr:
                                xmin=c
                            elif x<=c and x<=xnbr:
                                xmin=x
                            else:
                                xmin=xnbr
                            x1,y1=c,r
                            x2,y2=x,y
                            x3,y3=xnbr,ynbr
                            for py in range(ymin,ymax):
                                for px in range(xmin,xmax):
                                    denominator=(y2-y3)*(x1-x3)+(x3-x2)*(y1-y3)
                                    alpha=((y2-y3)*(px-x3)+(x3-x2)*(py-y3))/denominator
                                    beta=((y3-y1)*(px-x3)+(x1-x3)*(py-y3))/denominator
                                    gamma=1-alpha-beta
                                    if 0<=alpha<=1 and 0<=beta<=1 and 0<=gamma<=1:
                                        out_flood_img[py][px]=100
                    nbr_data[nbr_count]=y*ip_flood_img.shape[1]+x
                    nbr_count+=1

@cuda.jit
def quantize_img(img_array_d,img_quantized_d,quant_size):
    r,c=cuda.grid(2)
    #quant_size=int(256/ncolors)
    if r<img_array_d.shape[0] and c<img_array_d.shape[1]:
        color=img_array_d[r][c]
        color_quant=int(math.ceil(color/quant_size))*quant_size
        img_quantized_d[r][c]=color_quant

@cuda.jit
def get_color_count(img_array,cc_array):
    r,c=cuda.grid(2)
    # cc_array - color count array
    if r<img_array.shape[0] and c<img_array.shape[1]:
        cuda.atomic.add(cc_array,img_array[r][c],1)

@cuda.jit
def replace_color(img_array,out_img,color,threshold,new_color):
    r,c=cuda.grid(2)
    if r<img_array.shape[0] and c<img_array.shape[1]:
        if abs(img_array[r][c]-color)<=threshold:
            out_img[r][c]=new_color

def rotate_left(lst,n):
    return lst[n:]+lst[:n]

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

@cuda.jit
def find_next_ba(ba_max_pd_d,nz_ba_size_d,nz_ba_size_cum_d,bound_abstract_d,ba_threshold):
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
        n=nz_ba_d[ci]+1
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
                pd_max_i=n
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



def arrange_sign(sign_array):
    sign_list=sign_array.tolist()
    while 1:
        if sign_list.count(1)==len(sign_list) or sign_list.count(-1)==len(sign_list) or sign_list[0]!=sign_list[-1]:
            break
        sign_list.append(sign_list[0])
        sign_list.pop(0)
    return sign_list


@cuda.jit
def inverse_sign_cuda(array_list):
    ci=cuda.grid(1)
    if ci<len(array_list):
        array_list[ci]*=-1
    cuda.syncthreads()

def inverse_sign(string):
    inv=list()
    for i in string:
        if int(i)==0:
            inv.append("1")
        else:
            inv.append("0")
    return ''.join(inv)

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


@cuda.jit
def find_detail(nz_ba_d,bound_threshold_d,bound_abstract_d,bound_data_ordered_d,max_pd_d,scaled_shape,bound_mark_d,ba_size_d):
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
        n=nz_ba_d[ci]
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
                pd_max_i=n
            n+=1
        cuda.syncthreads()
        if pd_max>threshold:
            bound_abstract_d[pd_max_i]=pd_max_i
            seed_=bound_mark_d[nz_ba_d[ci]-1]
            #ba_size_d[seed_]+=1
            cuda.atomic.add(ba_size_d,seed_,1)

        
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
        init_n=n+1
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
def get_neighbor_data_init(nz_a,tmp_img,neighbor_data_d):
    # method to find connected neighbors
    ci=cuda.grid(1)
    if ci<nz_a.shape[0]:
        index=nz_a[ci]
        y=int(index/tmp_img.shape[1])
        x=index%tmp_img.shape[1]
        r=y
        c=x
        color=tmp_img[r][c]
        n=0
        if tmp_img[r-1][c]!=500 and tmp_img[r-1][c]!=color:
            neighbor_data_d[ci][n]=(r-1)*tmp_img.shape[1]+c
            n+=1
        if tmp_img[r][c+1]!=500 and tmp_img[r][c+1]!=color:
            neighbor_data_d[ci][n]=r*tmp_img.shape[1]+c+1
            n+=1
        if tmp_img[r+1][c]!=500 and tmp_img[r+1][c]!=color:
            neighbor_data_d[ci][n]=(r+1)*tmp_img.shape[1]+c
            n+=1
        if tmp_img[r][c-1]!=500 and tmp_img[r][c-1]!=color:
            neighbor_data_d[ci][n]=r*tmp_img.shape[1]+c-1
            n+=1

            
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
            
            # condition for neighborhoodness
            if tmp_img[r-1][c]!=500 and tmp_img[r-1][c]!=color:
                neighbor_data_d[ci][n]=(r-1)*tmp_img.shape[1]+c
                n+=1
            if tmp_img[r][c+1]!=500 and tmp_img[r][c+1]!=color:
                neighbor_data_d[ci][n]=r*tmp_img.shape[1]+c+1
                n+=1
            if tmp_img[r+1][c]!=500 and tmp_img[r+1][c]!=color:
                neighbor_data_d[ci][n]=(r+1)*tmp_img.shape[1]+c
                n+=1
            if tmp_img[r][c-1]!=500 and tmp_img[r][c-1]!=color:
                neighbor_data_d[ci][n]=r*tmp_img.shape[1]+c-1
                n+=1
            # end

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
def get_bound_cuda2(tmp_img,bound_len_low,bound_len_high,seed_map_d,bound_info):
    r,c=cuda.grid(2)
    # last=0,1,2,3 for n,e,s,w respectively
    if r>0 and r<tmp_img.shape[0]-1 and c>0 and c<tmp_img.shape[1]-1 and tmp_img[r][c]!=500:
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
def get_neighbor_cuda(neighbor_data_d,tmp_img,neighbor_info):
    cr,cc=cuda.grid(2)
    # last=0,1,2,3 for n,e,s,w respectively
    if cr<neighbor_data_d.shape[0] and cc<neighbor_data_d.shape[1]:
        index=neighbor_data_d[cr][cc]
        if index==0:
            return
        y=int(index/tmp_img.shape[1])
        x=index%tmp_img.shape[1]

        r=y
        c=x
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
        neighbor_info[cr][cc]=min_i
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
    if r>=0 and r<img_array.shape[0] and c>=0 and c<img_array.shape[1]:
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
def read_bound_cuda2(img,width,img_boundary_d):
    """ blob_dict={color: [[pixels],[pixels]]"""
    r,c=cuda.grid(2)
    threshold=-1000
    if r>=0 and r<img.shape[0]-width and c>=0 and c<img.shape[1]-width:
        if abs(img[r][c]-img[r][c+width])>threshold: # left ro right
            img_boundary_d[r][c]=abs(img[r][c]-img[r][c+width])
            #img_boundary_d[r][c+1]=img[r][c+1]
        
        if abs(img[r][c]-img[r+width][c])>threshold: # top to bottom
            img_boundary_d[r][c]=abs(img[r][c]-img[r+width][c])
            #img_boundary_d[r+1][c]=img[r+1][c]
        
        if abs(img[r][c]-img[r+width][c+width])>threshold: # diagonal
            img_boundary_d[r][c]=abs(img[r][c]-img[r+width][c+width])
            #img_boundary_d[r+1][c+1]=img[r+1][c+1]

        if abs(img[r+width][c]-img[r][c+width])>threshold: # diagonal
            img_boundary_d[r+width][c]=abs(img[r+width][c]-img[r][c+width])
            #img_boundary_d[r][c+1]=img[r][c+1]



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

def draw_pixels_cuda2(pixels,exclusions,invert,i,img):
    draw_pixels_cuda2_[pixels.shape[0],1](pixels,exclusions,invert,i,img)
    cuda.synchronize()

@cuda.jit
def draw_pixels_cuda2_(pixels,exclusions,invert,i,img):
    cc=cuda.grid(1)
    if cc<pixels.shape[0]:
        if invert:
            if exclusions[cc]<0:
                r=int(pixels[cc]/img.shape[1])
                c=pixels[cc]%img.shape[1]
                img[r][c]=i
        else:
            if exclusions[cc]>0:
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
