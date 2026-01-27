# How does edge detection work?<br>
<img src='https://github.com/ps-nithin/pyrebel/raw/5d4158b02ef84751191a6de62c1681d7f7776237/images/Figure_1.png'></img><br>
An image can be divided into rows. Each row consist of intensities of color at each point across the row. A graph / 2D image as shown above is plotted such that each point in the row is plotted along the x-axis and the intensities at each point in the row is plotted along the y-axis. In this way, each row in the image is represented as a 2D graph containing the position in the x-axis and its intensity in the y-axis. For eq. running the below code displays the 2D graph at each row of the input image.<br><br>
```python3 image_graph.py --input images/lotus.png```<br><br>
Similarly, running the below code displays the abstracted 2D graph for the input row and image as shown in the above image.<br><br>
```python3 image_graph_abs.py --input images/lotus.jpg --row 200 --edge_threshold 20```<br><br>
This 2D cross-section of each row of the image is formed along vertical, horizontal, and two diagonal axis of the input image and is abstracted and then put togather to obtain a 2D image containing the edges as shown in the below image. The white points represent convex abstract points whereas the grey points represent concave abstract points.<br><br>
<img src='https://github.com/ps-nithin/pyrebel/raw/b35c4dd590692e6bb67602f9a7427674ac4e0617/images/edges.png'></img><br><br>
The dark edges are joined with neighboring dark edges without intersecting with light edges and similarly, the light edges are joined with neighboring light edges without intersecting with dark edges to obtain a smooth edge image.<br>

For eq. running the below code gives a smooth edge as shown in the below image.<br>

```python3 pyrebel_main_edge_join.py --input images/lotus.jpg --edge_threshold 10```<br><br>
<img src='https://github.com/ps-nithin/pyrebel/raw/b35c4dd590692e6bb67602f9a7427674ac4e0617/images/output_merge.png'></img><br>
