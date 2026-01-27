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
from numba import cuda
from PIL import Image
import argparse,time,math
from pyrebel.preprocess import Preprocess
from pyrebel.abstract import Abstract
from pyrebel.edge import Edge
from pyrebel.learn import Learn
from pyrebel.utils import *


def paint_edges(img_array_rgb,edges,block_threshold,paint_threshold):
    img_array_rgb_d=cuda.to_device(img_array_rgb)
    edges_img_d=cuda.to_device(edges)
    block_img=img_array_rgb
    block_img_d=cuda.to_device(block_img)
    threadsperblock=(16,16)
    blockspergrid_x=math.ceil(edges.shape[0]/threadsperblock[0])
    blockspergrid_y=math.ceil(edges.shape[1]/threadsperblock[1])
    blockspergrid=(blockspergrid_x,blockspergrid_y)
    
    k=0
    while 1:
        n=block_threshold
        while 1:
            draw_blocks_edges[blockspergrid,threadsperblock](img_array_rgb_d,edges_img_d,block_img_d,n)
            cuda.synchronize()
            if n<2:
                break
            n-=1
            img_array_rgb_d=cuda.to_device(block_img_d.copy_to_host())
        k+=1
        print("painting",k,"/",paint_threshold,end='\r')
        if k==paint_threshold:
            print()
            break        
    block_img_h=block_img_d.copy_to_host()
    return block_img_h

@cuda.jit
def clone_image_rgb(img_array_orig,image_to_clone,img_cloned,color):
    """Draws pixels in 'image_to_clone' with color '255' to 'img_cloned' with the color of corresponding pixels in 'img_array_orig'"""
    
    r,c=cuda.grid(2)
    if r>0 and r<img_array_orig.shape[0] and c>0 and c<img_array_orig.shape[1]:
        if image_to_clone[r][c]==color:
            img_cloned[r][c][0]=img_array_orig[r][c][0]
            img_cloned[r][c][1]=img_array_orig[r][c][1]
            img_cloned[r][c][2]=img_array_orig[r][c][2]
            #cuda.atomic.add(count,0,1)

@cuda.jit
def join_edges(edges_d,out_image_d,threshold,color):
    r,c=cuda.grid(2)
    if threshold<r<edges_d.shape[0]-threshold-1 and threshold<c<edges_d.shape[1]-threshold-1 and edges_d[r][c]==color:
        #if edges_d[r-1][c]==edges_d[r+1][c]==color or edges_d[r][c-1]==edges_d[r][c+1]==color:
        #    return
        for rrr in range(r-threshold,r+threshold+1):
            for ccc in range(c-threshold,c+threshold+1):
                if edges_d[rrr][ccc]==color:
                    #if edges_d[rrr-1][ccc]==edges_d[rrr+1][ccc]==color or edges_d[rrr][ccc-1]==edges_d[rrr][ccc+1]==color:
                    #    return
                    found_wall=False
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
                        if cc==x and rr==y:
                            break
                        e2=2*err
                        if e2>-dy:
                            err-=dy
                            cc+=sx
                        elif e2<dx:
                            err+=dx
                            rr+=sy
                        if edges_d[rr][cc]!=0 and edges_d[rr][cc]!=color:
                            found_wall=True
                            break
                    if found_wall:
                        continue
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

def crop_center(img, crop_width, crop_height):
    """
    Crops the center of an image to the specified width and height.
    """
    width, height = img.size

    # Calculate coordinates for the center crop
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height

    # Crop the image
    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img
       
parser=argparse.ArgumentParser()
parser.add_argument("-i","--input",help="Input file name.")
parser.add_argument("-c","--camera",help="Input from camera.")
parser.add_argument("-et","--edge_threshold",help="Threshold of edge detection.")
parser.add_argument("-b","--block_threshold",help="Block threshold.")
parser.add_argument("-p","--paint_threshold",help="Paint threshold.")
parser.add_argument("-s","--scale_factor",help="Scaling factor.")
args=parser.parse_args()


if args.edge_threshold:
    edge_threshold=int(args.edge_threshold)
else:
    edge_threshold=10
if args.block_threshold:
    block_threshold=int(args.block_threshold)
else:
    block_threshold=20
if args.paint_threshold:
    paint_threshold=int(args.paint_threshold)
else:
    paint_threshold=5
if args.scale_factor:
    scale_factor=int(args.scale_factor)
else:
    scale_factor=1
    
def process_image(img_array_rgb_orig):
    start_time=time.time()
    img_array=np.array(Image.fromarray(img_array_rgb_orig).convert('L'))
    img_array_rgb_orig_d=cuda.to_device(img_array_rgb_orig)
    
    edge1=Edge(img_array)
    edge1.find_edges(edge_threshold)
    edges1=edge1.get_edges_both()
    edges_input_d=cuda.to_device(edges1)

    threadsperblock=(32,32)
    blockspergrid=(math.ceil(img_array.shape[0]/32),math.ceil(img_array.shape[1]/32))
    join_count=10
    n=1
    join_threshold=1
    while 1:
        out_image_d=cuda.to_device(edges_input_d.copy_to_host())
        join_edges[blockspergrid,threadsperblock](edges_input_d,out_image_d,join_threshold,100)
        cuda.synchronize()
        edges_input2_d=cuda.to_device(out_image_d.copy_to_host())    
        out_image2_d=cuda.to_device(edges_input2_d.copy_to_host())
        join_edges[blockspergrid,threadsperblock](edges_input2_d,out_image2_d,join_threshold,255)
        cuda.synchronize()
        edges_input_d=cuda.to_device(out_image2_d.copy_to_host())
        print("join count",n,"/",join_count,end='\r')
        if n==join_count:
            print()
            break
        n+=1
        join_threshold+=1

    edges_joined=out_image2_d.copy_to_host()
    
    edges_single=np.zeros(img_array.shape,dtype=np.uint8)
    edges_single_d=cuda.to_device(edges_single)
    clone_image[blockspergrid,threadsperblock](out_image2_d,edges_single_d,100)
    cuda.synchronize()
    edges_single_h=edges_single_d.copy_to_host()
    
    edges_rgb_d=cuda.to_device(np.full(img_array_rgb_orig.shape,0,dtype=np.uint8))
    #clone_image_rgb[blockspergrid,threadsperblock](img_array_rgb_orig_d,out_image2_d,edges_rgb_d,255)
    #cuda.synchronize()
    clone_image_rgb[blockspergrid,threadsperblock](img_array_rgb_orig_d,out_image2_d,edges_rgb_d,100)
    cuda.synchronize()

    edges_rgb_h=edges_rgb_d.copy_to_host()
    edges_painted=paint_edges(edges_rgb_h,edges_joined,block_threshold,5)
        
    # Save the output to disk.    
    Image.fromarray(edges_painted).convert('RGB').save("edges_painted.png")
    Image.fromarray(edges_joined).convert('RGB').save("output_joined.png")
    Image.fromarray(edges_single_h).convert('RGB').save("output.png")
    Image.fromarray(edges1).convert('RGB').save("edges.png")
    print("Finished in",time.time()-start_time,"seconds at",float(1/(time.time()-start_time)),"fps.")  

def process_input():
    while 1:
        img_array_rgb=np.array(Image.open(args.input).convert('RGB'))
        img_array_scaled=scale_camera_frame(img_array_rgb,scale_factor)
        process_image(img_array_scaled)  
        break
    
def process_camera_jutils():
    from jetson_utils import videoSource, videoOutput, Log
    from jetson_utils import cudaAllocMapped,cudaConvertColor
    from jetson_utils import cudaToNumpy,cudaDeviceSynchronize,cudaFromNumpy
    def convert_color(img,output_format):
        converted_img=cudaAllocMapped(width=img.width,height=img.height,
                format=output_format)
        cudaConvertColor(img,converted_img)
        return converted_img

    input_capture = videoSource("csi://0", options={'width':640,'height':640,'framerate':30,'flipMethod':'rotate-360'})
    #output = videoOutput("", argv=sys.argv)
    while 1:
        input_capture.Capture()
        img_array_rgb = input_capture.Capture()
        if img_array_rgb is None: # timeout
            print("No camera capture!")
            continue  
        img_rgb=cudaToNumpy(img_array_rgb)
        cudaDeviceSynchronize()
        img_array_scaled=scale_camera_frame(img_rgb,scale_factor)
        process_image(img_array_scaled)  
        
def scale_camera_frame(img_array_rgb_orig,scale_factor=1,crop_width=-1):
    #scale_factor=int(2048/max(img_array_rgb_orig.shape))
    print("scale=",scale_factor)
    if crop_width==-1:
        crop_width=img_array_rgb_orig.shape[1]
        crop_height=img_array_rgb_orig.shape[0]
    else:
        crop_height=crop_width
    #img_array_i=crop_center(Image.fromarray(img_array_rgb_orig).convert('L'),crop_width,crop_height)
    img_array_rgb_i=crop_center(Image.fromarray(img_array_rgb_orig),crop_width,crop_height)
    
    #img_array=np.array(img_array_i.resize((int(img_array_rgb_i.width*scale_factor),int(img_array_rgb_i.height*scale_factor)),Image.LANCZOS))
    img_array_rgb=np.array(img_array_rgb_i.resize((int(img_array_rgb_i.width*scale_factor),int(img_array_rgb_i.height*scale_factor)),Image.LANCZOS))
    return img_array_rgb

def process_camera_gst():
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    # Initialize GStreamer
    Gst.init(None)

    # GStreamer pipeline string for CSI camera
    csi_pipeline=" ! ".join([
        "nvarguscamerasrc sensor-id=0",  # Adjust sensor-id if multiple cameras
        "video/x-raw(memory:NVMM), width=(int)640, height=(int)640, format=(string)NV12, framerate=(fraction)30/1",
        "nvvidconv",
        "video/x-raw, format=(string)BGRx",
        "videoconvert",
        "video/x-raw, format=(string)RGB",
        "appsink name=sink emit-signals=true max-buffers=1 drop=true"
        ])
        
    # GStreamer pipeline string for USB camera
    usb_pipeline=" ! ".join([
        "v4l2src device=/dev/video0",
        "videoconvert",
        "video/x-raw, format=(string)RGB",
        "appsink name=sink emit-signals=true max-buffers=1 drop=true"
        ])
        
    # Create pipeline
    pipeline = Gst.parse_launch(usb_pipeline) # Choose pipeline string as required
    appsink = pipeline.get_by_name("sink")

    # Start the pipeline
    pipeline.set_state(Gst.State.PLAYING)

    # Main loop
    while 1:
        sample = appsink.emit("try-pull-sample", 100*Gst.MSECOND)
        if sample:
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            width = caps.get_structure(0).get_value("width")
            height = caps.get_structure(0).get_value("height")

            # Map buffer to numpy array
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                frame_data = np.frombuffer(map_info.data, dtype=np.uint8)
                frame = frame_data.reshape((height, width, 3))  # BGR format
                buffer.unmap(map_info)            
                Image.fromarray(frame).convert('RGB').save("camera.png")
                frame_scaled=scale_camera_frame(frame,scale_factor)
                process_image(frame_scaled)                
                # Now `frame` is a NumPy array (height, width, 3)
                #print(f"Captured frame: {frame.shape}")
            else:
                print("Buffer mapping failed.")

    # Clean up
    pipeline.set_state(Gst.State.NULL)
    print("Pipeline stopped.")

def main():
    if args.camera:
        process_camera_gst()
    elif args.input:
        process_input()
        
if __name__ == '__main__':
    main()
