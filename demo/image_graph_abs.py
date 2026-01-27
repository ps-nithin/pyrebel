import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse,math,time
from numba import cuda
from pyrebel.abstract import Abstract
from pyrebel.edge import Edge
from pyrebel.utils import *

parser=argparse.ArgumentParser()
parser.add_argument("-i","--input",help="Input filename.")
parser.add_argument("-r","--row",help="Row index.")
parser.add_argument("-t","--edge_threshold",help="Edge threshold.")
def main():
    args=parser.parse_args()
    if args.input:
        img_array=np.array(Image.open(args.input).convert('L'))
    else:
        print("No input filename provided.")
        return
    if args.row:
        row=int(args.row)
    else:
        # Defaults to the middle row of the image.
        row=int(img_array.shape[0]/2)
        
    if args.edge_threshold:
        edge_threshold=int(args.edge_threshold)
    else:
        edge_threshold=10
    print("Row=",row)
    img_array_d=cuda.to_device(img_array)
    edge=Edge(img_array)
    edge.find_edges(edge_threshold)
    edges_dark=edge.get_edges_one_original(0)
    edges_light=edge.get_edges_one_original(1)
    #Image.fromarray(edges_dark).convert('RGB').save('dark.png')
    #Image.fromarray(edges_light).convert('RGB').save('light.png')
    pos=list(range(0,img_array.shape[1]))
    pos_mark=[i for i in pos if edges_light[row][i]>0]
    pos_mark2=[i for i in pos if edges_dark[row][i]>0]
    y_mark=edges_light[row][pos_mark]
    y_mark2=edges_dark[row][pos_mark2]    
    
    plt.figure(figsize=(10,5))
    plt.plot(pos,img_array[row],color='lightgrey',label='Data points in a row')
    plt.plot(pos_mark,y_mark,'o',ms=5,color='white',markeredgecolor = 'black',label='Convex abstract points')
    plt.plot(pos_mark2,y_mark2,'o',ms=5,color='grey',markeredgecolor = 'black',label='Concave abstract points')
    plt.xlabel('Position')
    plt.ylabel('Intensity')
    plt.title('2D plot of Intensity for row='+str(row)+' with abstract points')
    plt.legend()
    plt.show()


if __name__=="__main__":
        main()
