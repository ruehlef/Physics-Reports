<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1 plus MathML 2.0//EN"
        "HTMLFiles/xhtml-math11-f.dtd">

<!-- Created with the Wolfram Language : www.wolfram.com -->

<html xmlns="http://www.w3.org/1999/xhtml">
<head>
 <link href="HTMLFiles/Mathematica_Example.css" rel="stylesheet" type="text/css" />
</head>

<body>
<a href="http://htmlpreview.github.com/?https://github.com/ruehlef/Physics-Reports/tree/master/Chapter_2/2.1/Mathematica_Example/Mathematica_Example.htm">To display this file properly, open it outside of Github.</a>
<br/><br/>

<p class="Section">
 Mathematica implementation of the simple NN that classifies bundle stability (cf.Section 2.1)
</p>



<p class="Text">
 Optional : Seed the random number generator for reproducibility
</p>



<p class="Input">
 <img src="HTMLFiles/Mathematica_Example_1.gif" alt="Mathematica_Example_1.gif" width="133" height="39" style="vertical-align:middle" />
</p>

<p class="Subsection">
 Read in the full data set
</p>



<p class="Input">
 <img src="HTMLFiles/Mathematica_Example_2.gif" alt="Mathematica_Example_2.gif" width="593" height="144" style="vertical-align:middle" />
</p>

<p class="Subsection">
 Perform a train:test split
</p>



<p class="Input">
 <img src="HTMLFiles/Mathematica_Example_3.png" alt="Mathematica_Example_3.png" width="580" height="207" style="vertical-align:middle" />
</p>

<p class="Subsection">
 Define the NN hyperparameters
</p>



<p class="Input">
 <img src="HTMLFiles/Mathematica_Example_4.png" alt="Mathematica_Example_4.png" width="276" height="207" style="vertical-align:middle" />
</p>

<p class="Subsection">
 Set up the NN
</p>



<p class="Input">
 <img src="HTMLFiles/Mathematica_Example_5.png" alt="Mathematica_Example_5.png" width="706" height="39" style="vertical-align:middle" />
</p>

<p class="Output">
 <img src="HTMLFiles/Mathematica_Example_6.gif" alt="Mathematica_Example_6.gif" width="338" height="60" style="vertical-align:middle" />
</p>

<p class="Subsection">
 Initialize the network
</p>



<p class="Input">
 <img src="HTMLFiles/Mathematica_Example_7.png" alt="Mathematica_Example_7.png" width="173" height="17" style="vertical-align:middle" />
</p>

<p class="Subsection">
 Train the NN
</p>



<p class="Input">
 <img src="HTMLFiles/Mathematica_Example_8.png" alt="Mathematica_Example_8.png" width="602" height="102" style="vertical-align:middle" />
</p>

<p class="Subsection">
 Plot the loss during training
</p>



<p class="Input">
 <img src="HTMLFiles/Mathematica_Example_9.png" alt="Mathematica_Example_9.png" width="55" height="17" style="vertical-align:middle" />
</p>

<p class="Output">
 <img src="HTMLFiles/Mathematica_Example_10.gif" alt="Mathematica_Example_10.gif" width="642" height="138" style="vertical-align:middle" />
</p>

<p class="Input">
 <img src="HTMLFiles/Mathematica_Example_11.gif" alt="Mathematica_Example_11.gif" width="599" height="39" style="vertical-align:middle" />
</p>

<p class="Output">
 <img src="HTMLFiles/Mathematica_Example_12.gif" alt="Mathematica_Example_12.gif" width="471" height="201" style="vertical-align:middle" />
</p>

<p class="Subsection">
 Evaluate the NN
</p>



<p class="Input">
 <img src="HTMLFiles/Mathematica_Example_13.gif" alt="Mathematica_Example_13.gif" width="655" height="333" style="vertical-align:middle" />
</p>

<p class="Print">
 <img src="HTMLFiles/Mathematica_Example_14.png" alt="Mathematica_Example_14.png" width="158" height="16" style="vertical-align:middle" />
</p>

<p class="Print">
 <img src="HTMLFiles/Mathematica_Example_15.png" alt="Mathematica_Example_15.png" width="134" height="1510" style="vertical-align:middle" />
</p>

<p class="Print">
 <img src="HTMLFiles/Mathematica_Example_16.png" alt="Mathematica_Example_16.png" width="144" height="16" style="vertical-align:middle" />
</p>

<p class="Subsection">
 Plot prediction of NN on all data
</p>



<p class="Input">
 <img src="HTMLFiles/Mathematica_Example_17.gif" alt="Mathematica_Example_17.gif" width="580" height="60" style="vertical-align:middle" />
</p>

<p class="Output">
 <img src="HTMLFiles/Mathematica_Example_18.gif" alt="Mathematica_Example_18.gif" width="360" height="290" style="vertical-align:middle" />
</p>

<p class="Input">
 <img src="HTMLFiles/Mathematica_Example_19.png" alt="Mathematica_Example_19.png" width="661" height="17" style="vertical-align:middle" />
</p>




<div style="font-family:Helvetica; font-size:11px; width:100%; border:1px none #999999; border-top-style:solid; padding-top:2px; margin-top:20px;">
 <a href="http://www.wolfram.com/language/" style="color:#000; text-decoration:none;">
  <span style="color:#555555">Created with the Wolfram Language</span> 
 </a>
</div>
</body>

</html>
