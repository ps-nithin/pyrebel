# pyrebel
# On Jetson Nano
Usage:<br>
```python3 pynvrebel.py --input <filename.png>```<br><br>
Optional arguments<br>
```--threshold <value>``` Change the threshold value. Defaults to 32.<br>
```--blob <value>``` Select blob to draw in output. Defaults to 0.<br>
```--layer <value>``` Select layer for which change is printed. Defaults to 0.<br>
```--output <output filename>```Change output filename. Defaults to "output.png".<br><br>
For example,<br>
```python3 pynvrebel.py --input letters.png --threshold 32```<br>

The expected input files are grayscale images of figures like <a href="https://github.com/ps-nithin/pyrebel/blob/main/letters.png">letters.png</a><br>

Using <a href="https://github.com/ps-nithin/pyrebel/blob/main/aaa.png">aaa.png</a> or <a href="https://github.com/ps-nithin/pyrebel/blob/main/sss.png">sss.png</a> demonstrates how the program responds to similar figures of different scales.<br>

Running the above program will open an image showing the abstract pixels in the input image file as white pixels and saves the output to disk.

# Read more about the logic implemented <a href="https://github.com/ps-nithin/pyrebel/blob/main/abstract.pdf">here</a>

# Let the data shine!
