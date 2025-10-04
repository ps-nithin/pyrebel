# How does the program learn and recognize patterns?<br>
Consider two images, the first image is to be learned and the second is to be recognized. <br>
<img src="sign-images/s.png"></img>&nbsp;1.<img src="sign-images/s-ripple.png"></img>&nbsp;2.
<br><br>
(a) The first step is to find the farthest points in the pattern. This divides the pattern into two and the points 1 and 2 are the first abstract points.<br>
<img src="sign-images/learn/s2.png"></img>&nbsp;3.<img src="sign-images/recognize/s-ripple2.png"></img>4.
<br><br>
(b) The next step is to find a point fartest from the line joining abstract points in the previous step. Thus, we obtain point 3.<br>
<img src="sign-images/learn/s3.png"></img>&nbsp;5.<img src="sign-images/recognize/s-ripple3.png"></img>6.
<br><br>
(c) The next is to find the signature, which is the concavity or convexity of each abstract points as you travel along the pattern in a specific order (here, clockwise direction).
Thus we get the following signatures.<br>
<img src="sign-images/learn/s3-sign-join.png"></img>&nbsp;7.<img src="sign-images/recognize/s-ripple3-sign-join.png"></img>
<br><br>
(c) The process is repeated till the farthest point is less than the threshold or the program reaches the specified number of layers. Here 10.<br>
# Layer 4
<img src="sign-images/learn/s4.png"></img>&nbsp;9.<img src="sign-images/recognize/s-ripple4.png"></img>
<br><br>
<img src="sign-images/learn/s4-sign-join.png"></img>&nbsp;11.<img src="sign-images/recognize/s-ripple4-sign-join.png"></img>
<br><br>
# Layer 5
<img src="sign-images/learn/s5.png"></img>&nbsp;13.<img src="sign-images/recognize/s-ripple5.png"></img>
<br><br>
<img src="sign-images/learn/s5-sign-join.png"></img>&nbsp;15.<img src="sign-images/recognize/s-ripple5-sign-join.png"></img>
<br><br>
# Layer 6
<img src="sign-images/learn/s6.png"></img>&nbsp;17.<img src="sign-images/recognize/s-ripple6.png"></img>
<br><br>
<img src="sign-images/learn/s6-sign-join.png"></img>&nbsp;19.<img src="sign-images/recognize/s-ripple6-sign-join.png"></img>
<br><br>
# Layer 7
<img src="sign-images/learn/s7.png"></img>&nbsp;21.<img src="sign-images/recognize/s-ripple7.png"></img>
<br><br>
<img src="sign-images/learn/s7-sign-join.png"></img>&nbsp;23.<img src="sign-images/recognize/s-ripple7-sign-join.png"></img>
<br><br>
# Layer 8
<img src="sign-images/learn/s8.png"></img>&nbsp;25.<img src="sign-images/recognize/s-ripple8.png"></img>
<br><br>
<img src="sign-images/learn/s8-sign-join.png"></img>&nbsp;27.<img src="sign-images/recognize/s-ripple8-sign-join.png"></img>
<br><br>
# Layer 9
<img src="sign-images/learn/s9.png"></img>&nbsp;29.<img src="sign-images/recognize/s-ripple9.png"></img>30.
<br><br>
<img src="sign-images/learn/s9-sign-join.png"></img>&nbsp;31.<img src="sign-images/recognize/s-ripple9-sign-join.png"></img>32.
<br><br>
# Layer 10
<img src="sign-images/learn/s10.png"></img>&nbsp;33.<img src="sign-images/recognize/s-ripple10.png"></img>34.
<br><br>
<img src="sign-images/learn/s10-sign-join.png"></img>&nbsp;35.<img src="sign-images/recognize/s-ripple10-sign-join.png"></img>36.
<br><br>
(d) The final step is to compare the signatures.<br>
<img src="sign-images/learn/sign-learn.png"></img>&nbsp;37.<img src="sign-images/recognize/recognize-sign.png"></img>38.
<br><br>
As you can see, the signatures obtained from recognition image (green) matches with the signatures obtained from the learning image (blue).(The signatures may be rotated).
Thus, the image is recognized.
