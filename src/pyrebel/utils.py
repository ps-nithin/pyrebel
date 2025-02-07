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
from math import sqrt

def draw_pixels_cuda(pixels,i,img):
    # Draws 'pixels' to 'img' with color 'i'
    draw_pixels_cuda_[pixels.shape[0],1](pixels,i,img)
    cuda.synchronize()

@cuda.jit
def draw_pixels_cuda_(pixels,i,img):
    # CUDA version of function 'draw_pixels_cuda()'
    cc=cuda.grid(1)
    if cc<pixels.shape[0]:
        r=int(pixels[cc]/img.shape[1])
        c=pixels[cc]%img.shape[1]
        img[r][c]=i

@cuda.jit
def increment_by_one(array_d):
    # Increments each item in device array 'array_d' by one. cuda version.
    ci=cuda.grid(1)
    if ci<len(array_d):
        array_d[ci]+=1
        cuda.syncthreads()

@cuda.jit
def decrement_by_one(array_d):
    # Decrements each item in device array 'array_d' by one. cuda version
    ci=cuda.grid(1)
    if ci<len(array_d):
        array_d[ci]-=1
        cuda.syncthreads()
        
def decrement_by_one_cuda(array):
    # Decrements each item in 'array' by one.
    array_d=cuda.to_device(array)
    decrement_by_one[len(array),1](array_d)
    cuda.synchronize()
    return array_d.copy_to_host()    

def draw_pixels_from_indices_cuda(indices,pixels,i,img):
    # Draws pixels from 'pixels' with indices 'indices' to image 'img' with color 'i'
    draw_pixels_from_indices_cuda_[indices.shape[0],1](indices,pixels,i,img)
    cuda.synchronize()

@cuda.jit
def draw_pixels_from_indices_cuda_(indices,pixels,i,img):
    # CUDA version of function 'draw_pixels_from_indices_cuda()'
    cc=cuda.grid(1)
    if cc<len(indices):
        r=int(pixels[indices[cc]]/img.shape[1])
        c=pixels[indices[cc]]%img.shape[1]
        img[r][c]=i
        
@cuda.jit
def image_to_wave(img_array_d,img_wave_pre_init_d):
    # Plot each row of 'img_array_d' in 2D space with color of pixels as y-coordinate.
    r,c=cuda.grid(2)
    if r<img_array_d.shape[0] and c<img_array_d.shape[1]:
        img_wave_pre_init_d[r*img_array_d.shape[1]+c]=img_array_d[r][c]*img_array_d.shape[1]+c

@cuda.jit
def init_abstract(img_array_d,bound_abstract_pre_d):
    ci=cuda.grid(1)
    if ci==0:
        bound_abstract_pre_d[ci]=ci+1
    elif ci<len(bound_abstract_pre_d) and ci%img_array_d.shape[1]==0:
        bound_abstract_pre_d[ci]=ci+1
        bound_abstract_pre_d[ci-1]=ci
    elif ci==len(bound_abstract_pre_d)-1:
        bound_abstract_pre_d[ci]=ci+1

@cuda.jit
def init_abstract_from_size(size_d,size_cum_d,init_abstract_d):
    ci=cuda.grid(1)
    if ci==0:
        init_abstract_d[0]=1
    elif ci>0 and ci<len(size_d):
        init_abstract_d[size_cum_d[ci]]=size_cum_d[ci]+1
        init_abstract_d[size_cum_d[ci]-1]=size_cum_d[ci]
    else:
        init_abstract_d[size_cum_d[len(size_d)-1]+size_d[len(size_d)-1]-1]=size_cum_d[len(size_d)-1]+size_d[len(size_d)-1]

@cuda.jit
def image_rotate45(img_array_d,img_rot45_d,img_rot45_mask_d):
    r,c=cuda.grid(2)
    if r<img_array_d.shape[0] and c<img_array_d.shape[1]:
        if r+c<img_array_d.shape[0]:
            img_rot45_d[r+c][c]=img_array_d[r][c]
            img_rot45_mask_d[r+c][c]=-1
        else:
            img_rot45_d[r+c][c-(r+c-img_array_d.shape[0])-1]=img_array_d[r][c]
            img_rot45_mask_d[r+c][c-(r+c-img_array_d.shape[0])-1]=-1

@cuda.jit
def image_rev_rotate45(img_rot45_d,img_rot45_mask_d,img_array_d):
    r,c=cuda.grid(2)
    if r<img_rot45_d.shape[0] and c<img_rot45_d.shape[1] and img_rot45_mask_d[r][c]!=-500:
        if r<img_array_d.shape[0]:
            img_array_d[r-c,c]=img_rot45_d[r][c]
        else:
            img_array_d[r-(r-(img_array_d.shape[0]-1))-c,r-(img_array_d.shape[0]-1)+c]=img_rot45_d[r][c]

@cuda.jit
def fill_column_zero(img_array_d):
    r,c=cuda.grid(2)
    if r<img_array_d.shape[0] and c<img_array_d.shape[1]:
        img_array_d[r][c]=img_array_d[r][0]

def draw_pixels_cuda2(pixels,exclusions,invert,i,img):
    # Draws 'pixels' to image 'img' with 'exclusions' with color 'i'
    draw_pixels_cuda2_[pixels.shape[0],1](pixels,exclusions,invert,i,img)
    cuda.synchronize()

@cuda.jit
def draw_pixels_cuda2_(pixels,exclusions,invert,i,img):
    # CUDA version of function 'draw_pixels_cuda2()'
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
 
@cuda.jit
def clone_image(img_array,img_clone,color):
    # draws pixels in 'img_array' with color 'color' to 'img_clone'
    r,c=cuda.grid(2)
    if r<img_array.shape[0] and c<img_array.shape[1]:
        if img_array[r][c]==color:
            img_clone[r][c]=color

@cuda.jit
def clone_image2(img_array_orig,image_to_clone,img_cloned,inv):
    # draws pixels in 'image_to_clone' with color '255' to 'img_cloned' with the color 
    # of corresponding pixels in 'img_array_orig'
    r,c=cuda.grid(2)
    if r>0 and r<img_array_orig.shape[0] and c>0 and c<img_array_orig.shape[1]:
        if image_to_clone[r][c]==255:
            if inv:
                img_cloned[r][c]=img_array_orig[r][c]
            else:
                img_cloned[r][c]=255-img_array_orig[r][c]
            #cuda.atomic.add(count,0,1)
  
@cuda.jit
def clean_quant_img(quant_img_d):
    r,c=cuda.grid(2)
    if r>0 and r<quant_img_d.shape[0]-1 and c>0 and c<quant_img_d.shape[1]-1:
        if quant_img_d[r][c]!=0 and quant_img_d[r-1][c-1]==0 and quant_img_d[r-1][c]==0 and quant_img_d[r-1][c+1]==0 and quant_img_d[r][c-1]==0 and quant_img_d[r][c+1]==0 and quant_img_d[r+1][c-1]==0 and quant_img_d[r+1][c]==0 and quant_img_d[r+1][c+1]==0:
            quant_img_d[r][c]=0
                      
@cuda.jit
def draw_lines_neighbors_all(img_array_d,neighbor_img_d,color,threshold):
    r,c=cuda.grid(2)
    if r>threshold and r<img_array_d.shape[0]-threshold and c>threshold and c<img_array_d.shape[1]-threshold and img_array_d[r][c]!=0:
        for rrr in range(r-threshold,r+threshold):
            for ccc in range(c-threshold,c+threshold):
                cur_dist=sqrt(pow(r-rrr,2)+pow(c-ccc,2))
                if img_array_d[r][c]==img_array_d[rrr][ccc] and cur_dist<threshold:
                    x=c
                    y=r
                    cc=ccc
                    rr=rrr
                    dx=abs(x-cc)
                    dy=abs(y-rr)
                    sx=1 if cc<x else -1
                    sy=1 if rr<y else -1
                    err=dx-dy
                    while True:
                        neighbor_img_d[rr][cc]=img_array_d[r][c]
                        if cc==x and rr==y:
                            break
                        e2=2*err
                        if e2>-dy:
                            err-=dy
                            cc+=sx
                        elif e2<dx:
                            err+=dx
                            rr+=sy

@cuda.jit
def winding_number_kernel(polygon,bound_data_ordered_d,img_array_d,winding_out_img_d):
    """
    GPU kernel to compute the winding number for each point.

    Parameters:
    points: Array of points to check (n_points x 2).
    polygon: Vertices of the polygon (n_vertices x 2).
    results: Output array for each point (n_points).
    """
    py,px = cuda.grid(2)
    if py<winding_out_img_d.shape[0] and px<winding_out_img_d.shape[1]:
        winding_number = 0
        
        for i in range(polygon.shape[0]):
            p1=bound_data_ordered_d[polygon[i]]
            x1=p1%winding_out_img_d.shape[1]
            y1=int(p1/winding_out_img_d.shape[1])
            p2=bound_data_ordered_d[polygon[(i + 1) % polygon.shape[0]]]  # Next vertex (wraps around)
            x2=p2%winding_out_img_d.shape[1]
            y2=int(p2/winding_out_img_d.shape[1])

            if y1 <= py:
                if y2 > py:  # Upward crossing
                    if (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1) > 0:
                        winding_number += 1
            else:
                if y2 <= py:  # Downward crossing
                    if (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1) < 0:
                        winding_number -= 1

        if winding_number:
            winding_out_img_d[py][px]=img_array_d[y1][x1]
        #results[idx] = winding_number != 0  # Inside if winding number is non-zero


@cuda.jit
def draw_lines(nz_ba_d,nz_ba_size_d,bound_data_ordered_d,bound_mark_d,out_image_d,color,min_size):
    ci=cuda.grid(1)
    if ci<len(nz_ba_d)-1:
        if (nz_ba_d[ci]+1)==nz_ba_d[ci+1] or nz_ba_size_d[bound_mark_d[nz_ba_d[ci]]]<min_size+1:
            return
        a=bound_data_ordered_d[nz_ba_d[ci]]
        b=bound_data_ordered_d[nz_ba_d[ci+1]]
        a1=int(a/out_image_d.shape[1])
        a2=a%out_image_d.shape[1]
        b1=int(b/out_image_d.shape[1])
        b2=b%out_image_d.shape[1]

        x=a2
        y=a1
        cc=b2
        rr=b1

        dx=abs(x-cc)
        dy=abs(y-rr)
        sx=1 if cc<x else -1
        sy=1 if rr<y else -1
        err=dx-dy
        while True:
            out_image_d[rr][cc]=color
            if cc==x and rr==y:
                break
            e2=2*err
            if e2>-dy:
                err-=dy
                cc+=sx
            elif e2<dx:
                err+=dx
                rr+=sy

@cuda.jit
def quantize_img(img_array_d,img_quantized_d,quant_size):
    r,c=cuda.grid(2)
    #quant_size=int(256/ncolors)
    if r<img_array_d.shape[0] and c<img_array_d.shape[1]:
        color=img_array_d[r][c]
        color_quant=int(round(color/quant_size))*quant_size
        img_quantized_d[r][c]=color_quant

@cuda.jit
def scale_down_pixels(orig_pixels_d,pixels_d,orig_shape_d,shape_d,scale_d):
    ci=cuda.grid(1)
    if ci<len(pixels_d):
        r_orig=int(orig_pixels_d[ci]/orig_shape_d[1])
        c_orig=orig_pixels_d[ci]%orig_shape_d[1]
        r=int(r_orig/scale_d)
        c=int(c_orig/scale_d)
        pixels_d[ci]=r*shape_d[1]+c
   
