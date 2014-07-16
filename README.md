opencv-resistor
===============

Analyses a picture of a resistor to determine its value using the coloured bands

resistor-otsu.py uses Otsu threshholding and Canny edge detection to isolate the resistor. Then it uses fitLine to draw a line long-ways through the resistor, onto a new mask image.
