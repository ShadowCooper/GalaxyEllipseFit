# Galaxy Ellipse Fit
## About
A script that allows you to fit ellipses to a galaxy and its polar structure. Sometimes.

## Usage
Minimal:
```bash
python ellipseFitting.py myFolderWithFitsFiles myDesiredOutputFolder
```

Optional flags:
--no-plot: Don't save a final plot, only the ellipse fitting results.
--no-tex: Don't use LaTeX formatting in the final plot. If you have issues with LaTex when running the script, this fixes it.
--fast: Make the script faster by only simplifying the contour after the final polar contour candidate is chosen. Not recommended, gives significantly worse results.

Example:
```bash
python ellipseFitting.py myFolderWithFitsFiles myDesiredOutputFolder --fast --no-tex
```

In "fitEllipse()", there is a commented-out block where it can use differential evolution instead of Nelder-Mead to fit the ellipses, if --fast is off. Differential evolution is the most reliable, but is much slower than Nelder-Mead.

It'll automatically re-create any subdirectories that were in the input folder in the output folder

## Output
.json file with ellipse fit results, .jpg showing the fits and contours, (both of those are if the fitting was "successful") error log file for failed galaxies
