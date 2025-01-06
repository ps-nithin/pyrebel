# pyrebel
# Image abstraction demo

Usage:<br>
```python3 pyrebel_main.py --input <filename.png>```<br><br>
Optional arguments<br>
```--threshold <value>``` Selects the threshold of abstraction. Defaults to 5.<br><br>
For example,<br>
```python3 pyrebel_main.py --input images/abc.png --threshold 10```<br>
The output is written to 'output.png'
# Layers of abstraction
<img src="animation.gif"></img>

# Edge detection demo
This is a demo of edge detection achieved using data abstraction.<br>
```python3 pyrebel_main_edge.py --input <filename>```<br>

For eq.
```python3 pyrebel_main_edge.py --input images/wildlife.jpg```<br>
The output is written to 'output.png'

Running the above program will show the edges in the image.<br>
<img src="images/small_wildlife.jpg"></img><br>Below is the output image<br><img src="images/output_wildlife.png"></img>

At layer zero the most abstract details in the figure is compared. As we move into deeper layers finer details are compared which gives distinctiveness to each figure.
# Read more about the methods <a href="https://github.com/ps-nithin/pyrebel/blob/main/intro-r2.pdf">here</a>

# Let the data shine!
