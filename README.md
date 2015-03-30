opencv-resistor
===============

Analyses a picture of a resistor to determine its value using the coloured bands

Still very much a work in progress (it doesn't work yet). If there are
reflections in the original image (e.g. gold bands) it appears as multiple
different colours.

resistor-otsu.py uses Otsu threshholding and Canny edge detection to isolate the resistor. Then it uses fitLine to draw a line long-ways through the resistor, onto a new mask image.
