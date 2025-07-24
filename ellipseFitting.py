import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import os
import json
from tqdm import tqdm
import argparse

from astropy.io import fits
from astropy.visualization import ImageNormalize, HistEqStretch
from skimage.measure import find_contours
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist
from shapely.geometry import Polygon, Point
from shapely.affinity import scale, rotate, translate

# --- Constants ---
M0 = 22.5
PIXEL_SCALE = 0.262
GALAXY_SB = 25.0
ERROR_LOG_FILE = 'ellipseFittingErrors.txt'

plt.rcParams['text.usetex'] = True


def calculateIsophoteIntensity(targetSb, m0, pixelScale):
    """Calculates the pixel intensity corresponding to a target surface brightness."""
    return (pixelScale ** 2) * 10 ** ((m0 - targetSb) / 2.5)


def smoothContour(contour, sigma=13.0):
    """Smoothes a contour's (x, y) coordinates using a Gaussian filter."""
    ySmooth = gaussian_filter1d(contour[:, 0], sigma=sigma, mode='wrap')
    xSmooth = gaussian_filter1d(contour[:, 1], sigma=sigma, mode='wrap')
    return np.column_stack((ySmooth, xSmooth))


def log_errors(debug_dict, error_file, basename):
    """
    Prints errors to the console and appends them to the specified error log file.
    This function is called only when a fatal error occurs for a given file.
    """
    if not debug_dict:
        return

    # --- Prepare content for both console and file ---
    log_content = [f"------{basename}------"]
    # Sort keys to ensure a consistent order: numbers first, then strings.
    sorted_keys = sorted(debug_dict.keys(), key=lambda k: (isinstance(k, str), k))

    for key in sorted_keys:
        messages = debug_dict[key]
        for msg in messages:
            if key == GALAXY_SB:
                line = f"Galaxy Ellipse ({key}): {msg}"
            elif isinstance(key, (float, int)):
                line = f"{key:.1f}: {msg}"
            else:  # For general, non-SB level errors (e.g., key='Fatal')
                line = f"{msg}"
            log_content.append(line)

    # --- Print to console ---
    print("\n" + "\n".join(log_content))

    # --- Append to file ---
    # Use a double newline to separate entries from different files.
    with open(error_file, 'a') as f:
        f.write("\n".join(log_content) + "\n\n")


def fitEllipse(contour, polarBounds=False, galAngle=None):
    """
    Fits an ellipse to a contour by minimizing the radial distance.
    """
    if contour.shape[0] < 20:
        return None

    ycGuess, xcGuess = contour.mean(axis=0)
    xCoords, yCoords = contour[:, 1] - xcGuess, contour[:, 0] - ycGuess
    mxx, myy, mxy = np.mean(xCoords ** 2), np.mean(yCoords ** 2), np.mean(xCoords * yCoords)

    if (mxx - myy) ** 2 + 4 * mxy ** 2 < 0:
        return None

    commonTerm = np.sqrt((mxx - myy) ** 2 + 4 * mxy ** 2)
    smaGuess = np.sqrt(2 * (mxx + myy + commonTerm))
    smiGuess = np.sqrt(2 * (mxx + myy - commonTerm))
    angleGuessRad = 0.5 * np.arctan2(2 * mxy, mxx - myy)
    initialGuess = [xcGuess, ycGuess, smaGuess, smiGuess, angleGuessRad]

    bounds = [(-1e99, 1e99)] * 5
    if polarBounds:
        bounds[-1] = ((galAngle - 110) * np.pi / 180, (galAngle - 70) * np.pi / 180)

    def ellipse_radial_distance_objective(params, points):
        xc, yc, sma, smi, angleRad = params
        if sma <= 0 or smi <= 0 or smi > sma:
            return 1e12

        cos_a, sin_a = np.cos(angleRad), np.sin(angleRad)
        x_p = (points[:, 1] - xc) * cos_a + (points[:, 0] - yc) * sin_a
        y_p = -(points[:, 1] - xc) * sin_a + (points[:, 0] - yc) * cos_a

        point_angles = np.arctan2(y_p, x_p)

        cos_pa, sin_pa = np.cos(point_angles), np.sin(point_angles)
        if sma == 0 or smi == 0: return 1e12
        ellipse_radii = (sma * smi) / np.sqrt((smi * cos_pa) ** 2 + (sma * sin_pa) ** 2)

        point_radii = np.hypot(x_p, y_p)

        return np.sum((point_radii - ellipse_radii) ** 2)

    result = minimize(ellipse_radial_distance_objective, initialGuess, args=(contour,),
                      method='Nelder-Mead')  # don't use bfgs... nelder-mead is slow but gradient-free. had issues with bfgs.
    xc, yc, sma, smi, angleRad = result.x

    if smi > sma:
        sma, smi = smi, sma
        angleRad += np.pi / 2
    angleDeg = np.rad2deg(angleRad) % 180.0

    return {'xc': xc, 'yc': yc, 'sma': abs(sma), 'smi': abs(smi),
            'angleDeg': angleDeg, 'contour': contour,
            'fit_error': result.fun}


def simplify_contour(contour, max_dist):
    """
    Removes relatively small, circuitous loops from a closed contour.

    The algorithm iterates through each point and finds potential shortcuts to
    other points that are nearby in space but far away along the contour.
    It prioritizes closing the most "circuitous" loops.

    Args:
        contour (np.ndarray): A NumPy array of shape (N, 2) representing the
                              points of the contour, e.g., [[x1, y1], [x2, y2], ...].
                              Can be either (x, y) coordinates or (y, x) coordinates;
                              the returned contour will match the system used.
        max_dist (float): The maximum Euclidean distance to consider for a shortcut.
                          Recommended: 0.4 * SMI (semi-major axis) of the galaxy
                          ellipse.

    Returns:
        np.ndarray: The simplified contour as a NumPy array.
    """
    # 1. --- INITIAL SETUP ---
    # Work with a Python list of lists for efficient modification
    work_contour = contour.tolist()

    # If the contour is closed (first point == last point), remove duplicate
    if work_contour[0] == work_contour[-1]:
        work_contour.pop()

    if not work_contour:
        return np.array([])

    est_spacing = np.linalg.norm(np.array(work_contour)[1] - np.array(work_contour)[0])
    min_step = 1.5 * (max_dist / est_spacing)  # what defines a "long enough" loop? If a point is within max_dist, but is >= 1.5 times greater contour distance than the contour distance that corresponds to a straight path away from the given point, then that loop should be removed.
    # Ensure contour is valid and sufficiently long
    if contour is None or len(contour) < min_step:
        return contour

    # 2. --- PRE-PROCESSING: RE-ORDER CONTOUR ---
    # Start the contour at the point farthest from the origin (0, 0)
    # Use NumPy for this vectorized calculation
    initial_points = np.array(work_contour)
    # We can use squared distance to avoid a sqrt calculation
    farthest_idx = np.argmax(np.sum(initial_points ** 2, axis=1))

    # Roll the contour so the farthest point is at index 0
    work_contour = np.roll(initial_points, -farthest_idx, axis=0).tolist()

    # 3. --- MAIN SIMPLIFICATION LOOP ---
    i = 0
    while i < len(work_contour):
        n = len(work_contour)
        # For the loop to terminate, we need to check this condition again
        if n < min_step:
            break

        # Use NumPy for fast, vectorized distance calculations
        current_point = np.array(work_contour[i])
        all_points = np.array(work_contour)

        # 1. Calculate all distances at once (Euclidean and Contour)
        euclidean_distances = np.linalg.norm(all_points - current_point, axis=1)
        all_indices = np.arange(n)
        contour_distances = (all_indices - i + n) % n

        # 2. Create a boolean mask to find all valid candidates simultaneously
        #    This replaces the 'if' statement inside the loop.
        mask = (
                (contour_distances >= min_step) &
                (euclidean_distances <= max_dist) &
                (contour_distances <= 10 * max_dist / est_spacing) &
                (euclidean_distances > 1e-9)  # Avoid division by zero
        )

        # 3. Get the indices and data for only the valid candidates
        candidate_indices = all_indices[mask]

        # If there are any candidates, calculate their ratios and build the list
        candidates = []
        if candidate_indices.size > 0:
            candidate_contour_dists = contour_distances[mask]
            candidate_euclidean_dists = euclidean_distances[mask]

            # 4. Calculate all ratios at once
            ratios = candidate_contour_dists / candidate_euclidean_dists

            # 5. Build the final list of dictionaries using a fast list comprehension
            candidates = [
                {"ratio": r, "j": j, "dist": d}
                for r, j, d in zip(ratios, candidate_indices, candidate_euclidean_dists)
            ]
        # If we found any valid shortcuts, process the best one
        if candidates:
            # The best shortcut is the one with the highest ratio
            best_candidate = max(candidates, key=lambda x: x["ratio"])
            j_to_connect = best_candidate["j"]
            connection_dist = best_candidate["dist"]

            # --- Perform linear interpolation to fill the new shortcut ---
            start_point = current_point
            end_point = all_points[j_to_connect]

            # Estimate existing point spacing to maintain density.
            # We use the distance between the current point and the previous one, calculated before.

            # Avoid division by zero if points are identical
            if est_spacing < 1e-9: est_spacing = 1.0

            num_new_points = int(np.ceil(connection_dist / est_spacing - 1))

            new_points = []
            if num_new_points > 0:
                segment_vector = end_point - start_point
                # Generate N evenly spaced points along the new line segment
                for k in range(1, num_new_points + 1):
                    step = k / (num_new_points + 1)
                    new_pt = start_point + step * segment_vector
                    new_points.append(new_pt.tolist())

            # --- Rebuild the contour list (the efficient way) ---
            indices_to_remove = set()
            # This loop correctly identifies the indices on the "short path"
            # between i and j, handling wraparound.
            k = (i + 1) % n
            while k != j_to_connect:
                indices_to_remove.add(k)
                k = (k + 1) % n

            # Build a new list containing only the points we want to keep
            temp_contour = [p for idx, p in enumerate(work_contour) if idx not in indices_to_remove]

            # Find the new index of our anchor point 'i' in the temporary list
            # We must use the original list object to find its new home
            i = temp_contour.index(work_contour[i])

            # Finally, insert the new interpolated points into the contour
            # This creates the final list for this iteration
            work_contour = temp_contour[:i + 1] + new_points + temp_contour[i + 1:]
        i += 1
    work_contour.append(work_contour[0])
    return np.array(work_contour)


def createShapelyEllipse(ellipseParams):
    """Creates a Shapely Polygon representation of an ellipse."""
    xc, yc, sma, smi, angleDeg = (ellipseParams['xc'], ellipseParams['yc'],
                                  ellipseParams['sma'], ellipseParams['smi'],
                                  ellipseParams['angleDeg'])
    circ = Point(0, 0).buffer(1)
    ell = scale(circ, xfact=sma, yfact=smi)
    ell = rotate(ell, angleDeg, origin=(0, 0), use_radians=False)
    ell = translate(ell, xoff=xc, yoff=yc)
    return ell


def analyzePolarGalaxy(fitsFilePath, outputDirectory, error_log_path, show_plot=True):
    """Main analysis function with contour dissection and advanced filtering."""
    print(f"--- Analyzing {os.path.basename(fitsFilePath)} ---")
    baseName = os.path.basename(fitsFilePath)  # Keep .fits for logging clarity

    # This dictionary will store all errors (fatal and non-fatal) for this file.
    # It will only be logged if a fatal error occurs.
    debug_log = {}

    def add_error(key, message):
        """Helper to add messages to the debug log, allowing multiple errors per key."""
        if key not in debug_log:
            debug_log[key] = []
        debug_log[key].append(message)

    try:
        with fits.open(fitsFilePath) as hdul:
            imageHdu = next((hdu for hdu in hdul if hdu.data is not None), None)
            if imageHdu is None:
                # FATAL ERROR 1: No image data.
                add_error('Fatal', 'No image data found in FITS file.')
                log_errors(debug_log, error_log_path, baseName)
                return
            data = imageHdu.data.astype(np.float32)
    except Exception as e:
        add_error('Fatal', f'Could not open or read FITS file: {e}')
        log_errors(debug_log, error_log_path, baseName)
        return

    junctionSmoothingSigma = 13.0
    imgCenterY, imgCenterX = data.shape[0] / 2.0, data.shape[1] / 2.0

    invertedData = np.max(data) - data
    norm = ImageNormalize(stretch=HistEqStretch(invertedData))

    # --- 1. Fit the Main Galaxy ---
    print(f"\nFitting galaxy at the {GALAXY_SB} mag/arcsec^2 isophote.")
    intensityGal = calculateIsophoteIntensity(GALAXY_SB, M0, PIXEL_SCALE)
    contoursGal = find_contours(data, level=intensityGal)
    centerContourGal = max([c for c in contoursGal if len(c) > 100 and
                            c[:, 1].min() < imgCenterX < c[:, 1].max() and
                            c[:, 0].min() < imgCenterY < c[:, 0].max()],
                           key=lambda c: Polygon(c).area, default=None)
    if centerContourGal is None:
        # FATAL ERROR 2: No galaxy contour.
        add_error(GALAXY_SB, 'Could not find galaxy contour.')
        log_errors(debug_log, error_log_path, baseName)
        return

    galaxyEllipse = fitEllipse(smoothContour(centerContourGal))
    if not galaxyEllipse:
        # FATAL ERROR 3: Failed to fit galaxy.
        add_error(GALAXY_SB, 'Failed to fit ellipse to main galaxy.')
        log_errors(debug_log, error_log_path, baseName)
        return

    galaxyContourPoly = Polygon(galaxyEllipse['contour'])
    galaxyShapelyEllipse = createShapelyEllipse(galaxyEllipse)

    # --- 2. Comprehensive Search for the Best Polar Structure Candidate ---
    print("Searching all isophotes for the best polar structure candidate...")
    good_candidates = []

    gal_xc, gal_yc, galAngle = galaxyEllipse['xc'], galaxyEllipse['yc'], galaxyEllipse['angleDeg']

    for sbLevel in np.arange(25.5, 27.5, 0.1):
        sbLevel = round(sbLevel, 1)
        intensity = calculateIsophoteIntensity(sbLevel, M0, PIXEL_SCALE)

        potentialPeanuts = [c for c in find_contours(data, level=intensity) if len(c) > 200]
        for peanutContour in potentialPeanuts:
            try:
                if not Polygon(peanutContour).contains(galaxyContourPoly.representative_point()):
                    continue
            except:
                continue

            # --- Quadrant-based junction finding with pre-filtering ---
            smoothedPeanut = smoothContour(peanutContour, sigma=junctionSmoothingSigma)
            simplified = simplify_contour(smoothedPeanut, 0.4 * galaxyEllipse['smi'])

            def getJunctions(contour, galEllipse, minAngle, maxAngle):
                angles = np.arctan2(contour[:, 0] - gal_yc, contour[:, 1] - gal_xc)
                angles = (angles + 2 * np.pi) % (2 * np.pi)
                angles = angles * 180 / np.pi
                goodPoints = [(point, idx) for idx, (point, angle) in enumerate(zip(contour, angles)) if
                              (angle - minAngle) % 360 <= (maxAngle - minAngle) % 360]
                if len(goodPoints) == 0:
                    return None, None, None
                dists = np.array([(galEllipse.distance(Point(point[::-1])), idx) for point, idx in goodPoints])
                farthestPoint = max(dists, key=lambda x: x[0])
                n = len(contour)

                def pointerCalc(pointer, change):
                    while galEllipse.distance(Point(contour[int(pointer)][::-1])) > 0.3 * farthestPoint[0] and min(
                            (pointer - farthestPoint[1] + n) % n, (farthestPoint[1] - pointer + n) % n) < int(n / 4):
                        pointer = (pointer + change) % n
                    return pointer

                pointer1 = pointer2 = farthestPoint[1]
                pointer1 = pointerCalc(pointer1, 1)
                pointer2 = pointerCalc(pointer2, -1)
                if min((pointer1 - farthestPoint[1] + n) % n, (farthestPoint[1] - pointer1 + n) % n) >= int(n / 4):
                    pointer1 = None
                if min((pointer2 - farthestPoint[1] + n) % n, (farthestPoint[1] - pointer2 + n) % n) >= int(n / 4):
                    pointer2 = None
                return pointer1, pointer2, farthestPoint

            junction1, junction2, farthestPoint1 = getJunctions(simplified, galaxyShapelyEllipse,
                                                                (galAngle - 135) % 360, (galAngle - 45) % 360)
            junction3, junction4, farthestPoint2 = getJunctions(simplified, galaxyShapelyEllipse,
                                                                (galAngle + 45) % 360, (galAngle + 135) % 360)
            junctionIndices = [junction1, junction2, junction3, junction4]

            if any(x is None for x in junctionIndices):
                # NON-FATAL ERROR 4: Couldn't find junctions for this SB level.
                add_error(sbLevel, "Could not find all 4 junction points.")
                continue

            junctionIndices = [int(junctionIndex) for junctionIndex in junctionIndices]
            junctionIndices = sorted(junctionIndices)
            simplifiedOriginalIndices = [(x, idx) for idx, x in enumerate(simplified)]

            arcs = []
            for i in range(len(junctionIndices)):
                start, end = junctionIndices[i], junctionIndices[(i + 1) % len(junctionIndices)]
                arc = simplifiedOriginalIndices[start:end] if end > start else simplifiedOriginalIndices[
                                                                               start:] + simplifiedOriginalIndices[:end]
                arcs.append(arc)

            polarArcs, galaxyArcs = [], []
            for arc in arcs:
                indices = {idx for _, idx in arc}
                if farthestPoint1[1] in indices or farthestPoint2[1] in indices:
                    polarArcs.append(np.array([pt for pt, _ in arc]))
                else:
                    galaxyArcs.append(np.array([pt for pt, _ in arc]))

            if not polarArcs: continue  # Should not happen if junctions are found, but good practice.

            polarEllipse = fitEllipse(np.vstack(polarArcs))
            if not polarEllipse:
                # NON-FATAL ERROR 5: Bad fit for this SB level.
                add_error(sbLevel, 'Could not fit to a processed contour.')
                continue

            if polarEllipse['sma'] > max(data.shape): continue
            if not createShapelyEllipse(polarEllipse).contains(Point(imgCenterX, imgCenterY)): continue

            angleBetween = min(abs(galaxyEllipse['angleDeg'] - polarEllipse['angleDeg']),
                               180 - abs(galaxyEllipse['angleDeg'] - polarEllipse['angleDeg']))
            if not (70 <= angleBetween <= 110):
                # NON-FATAL ERROR 6: Angle out of range for this SB level.
                add_error(sbLevel, "Angle between galaxy and polar ellipses wasn't between 70 and 110 degrees.")
                continue

            polarEllipse['sbLevelFound'] = sbLevel
            polarEllipse['galaxyArcs'] = galaxyArcs
            polarEllipse['polarArcs'] = polarArcs
            polarEllipse['junctionIndices'] = junctionIndices
            polarEllipse['smoothedPeanut'] = simplified
            good_candidates.append(polarEllipse)

    if not good_candidates:
        # FATAL ERROR 7: No candidates found after checking all isophotes.
        # This is a fatal error, so we add it to the log.
        add_error('Fatal', 'Analysis failed: After searching all isophotes, no candidate passed all filters.')
        # Now log all collected errors (fatal and non-fatal) for this file.
        log_errors(debug_log, error_log_path, baseName)
        return

    # If we reach here, the analysis was successful.
    # The baseName for output files should not have the .fits extension.
    outputBaseName = os.path.basename(fitsFilePath).replace('.fits', '')

    bestPolarFit = min(good_candidates, key=lambda e: e['fit_error'] / len(e['contour']) / e['sma'])

    # --- 3. Final Results and JSON Output ---
    angleGal, smaGal, smiGal = galaxyEllipse['angleDeg'], galaxyEllipse['sma'], galaxyEllipse['smi']
    anglePolar, smaPolar, smiPolar = bestPolarFit['angleDeg'], bestPolarFit['sma'], bestPolarFit['smi']
    isophoteFoundAt = bestPolarFit['sbLevelFound']
    angleBetween = min(abs(angleGal - anglePolar), 180 - abs(angleGal - anglePolar))

    helpString = "Angles are in degrees, measured counter-clockwise from the positive x-axis (horizontal). " \
                 "SMA/SMI (semi-major/minor axes) are in arcseconds. " \
                 "Isophotes are in mag/arcsec^2."
    resultsDict = {
        'galaxy': {'angle': round(angleGal, 2), 'sma': round(smaGal * PIXEL_SCALE, 2),
                   'smi': round(smiGal * PIXEL_SCALE, 2), 'isophote': GALAXY_SB},
        'polarStructure': {'angle': round(anglePolar, 2), 'sma': round(smaPolar * PIXEL_SCALE, 2),
                           'smi': round(smiPolar * PIXEL_SCALE, 2), 'isophote': round(isophoteFoundAt, 1)},
        'angleDiff': round(angleBetween, 2),
        'help': helpString
    }
    jsonPath = os.path.join(outputDirectory, f"{outputBaseName}_results.json")
    with open(jsonPath, 'w') as f:
        json.dump(resultsDict, f, indent=4)
    print(f"\nSaved results to {jsonPath}")

    print("\n--- Results ---")
    print(f"Galaxy Angle: {angleGal:.2f}°, SMA: {smaGal:.2f} pix, SMI: {smiGal:.2f} pix")
    print(
        f"Polar Angle:  {anglePolar:.2f}°, SMA: {smaPolar:.2f} pix, SMI: {smiPolar:.2f} pix (from system isophote at {isophoteFoundAt:.1f} mag/arcsec^2)")
    print(f"Angle between structures: {angleBetween:.2f}°")
    print("----------------\n")

    # --- 4. Final Visualization ---
    if show_plot:
        print("Generating final ellipse fit plot...")
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(norm(invertedData), origin='lower', cmap='gray')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

        eGal = Ellipse(xy=(galaxyEllipse['xc'], galaxyEllipse['yc']), width=2 * smaGal, height=2 * smiGal,
                       angle=angleGal, edgecolor='cyan', facecolor='none', lw=2,
                       label=rf'Galaxy Fit (SB = {GALAXY_SB} $\frac{{\mathrm{{mag}}}}{{{{\mathrm{{arcsec}}}}^2}}$)')
        ax.add_patch(eGal)

        ePolar = Ellipse(xy=(bestPolarFit['xc'], bestPolarFit['yc']), width=2 * smaPolar, height=2 * smiPolar,
                         angle=anglePolar, edgecolor='magenta', facecolor='none', lw=2,
                         label=rf'Polar Fit (SB = {isophoteFoundAt} $\frac{{\mathrm{{mag}}}}{{{{\mathrm{{arcsec}}}}^2}}$)')
        ax.add_patch(ePolar)

        galaxyArcs = bestPolarFit['galaxyArcs']
        polarArcs = bestPolarFit['polarArcs']

        ax.plot(galaxyEllipse['contour'][:, 1], galaxyEllipse['contour'][:, 0], color='cyan', linestyle=':', lw=1,
                alpha=0.7)
        ax.plot(polarArcs[0][:, 1], polarArcs[0][:, 0], color='magenta', linestyle=':', lw=1, alpha=0.7,
                label='Simplified Polar Arcs (Used)')
        ax.plot(polarArcs[1][:, 1], polarArcs[1][:, 0], color='magenta', linestyle=':', lw=1)
        ax.plot(galaxyArcs[0][:, 1], galaxyArcs[0][:, 0], color='lime', linestyle=':', lw=1.5, alpha=0.9,
                label='Galaxy Arcs (Unused)')
        ax.plot(galaxyArcs[1][:, 1], galaxyArcs[1][:, 0], color='lime', linestyle=':', lw=1.5, alpha=0.9)

        smoothedPeanutPlot = bestPolarFit['smoothedPeanut']
        junctionIndicesPlot = bestPolarFit['junctionIndices']
        ax.scatter(smoothedPeanutPlot[junctionIndicesPlot, 1], smoothedPeanutPlot[junctionIndicesPlot, 0], c='red',
                   s=40, zorder=10, label='Junctions')

        ax.set_title(f"Ellipse Fits for {outputBaseName}")
        ax.legend()
        finalPlotPath = os.path.join(outputDirectory, f"{outputBaseName}_ellipse_fit.jpg")
        fig.savefig(finalPlotPath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved final plot to {finalPlotPath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fit ellipses to a galaxy and its polar structure.")
    parser.add_argument("inputPath", help="Path to a single FITS file or a top-level directory to search.")
    parser.add_argument("outputDirectory", help="Path to the top-level directory where results will be saved.")
    parser.add_argument("--no-plot", action="store_true", help="Suppress saving the final JPG plots.")
    args = parser.parse_args()

    inputPath = args.inputPath
    outputBaseDir = args.outputDirectory
    showPlotFlag = not args.no_plot

    if not os.path.exists(outputBaseDir):
        os.makedirs(outputBaseDir, exist_ok=True)

    # Delete the error log file if it exists from a previous run.
    # The file will only be (re)created if an error occurs.
    if os.path.exists(ERROR_LOG_FILE):
        try:
            os.remove(ERROR_LOG_FILE)
        except OSError as e:
            print(f"Error: Could not remove existing log file '{ERROR_LOG_FILE}': {e}")

    filesToProcess = []
    if os.path.isfile(inputPath):
        if inputPath.endswith('.fits') and not inputPath.endswith('_mask.fits'):
            filesToProcess.append(inputPath)
    elif os.path.isdir(inputPath):
        print("Scanning for FITS files...")
        for root, dirs, files in os.walk(inputPath):
            baseNames = {f.replace('_masked.fits', '').replace('.fits', '') for f in files if
                         f.endswith('.fits') and not f.endswith('_mask.fits')}
            for baseName in baseNames:
                regularPath = os.path.join(root, f"{baseName}.fits")
                maskedPath = os.path.join(root, f"{baseName}_masked.fits")
                if os.path.exists(regularPath):
                    filesToProcess.append(regularPath)
                elif os.path.exists(maskedPath):
                    filesToProcess.append(maskedPath)

    if not filesToProcess:
        print(f"No processable '.fits' files found in '{inputPath}'.")
    else:
        print(f"Found {len(filesToProcess)} files to process.")
        for fitsFilePath in tqdm(filesToProcess, desc="Processing Galaxies"):
            try:
                if os.path.isdir(inputPath):
                    relativePath = os.path.relpath(os.path.dirname(fitsFilePath), inputPath)
                    outputDir = os.path.join(outputBaseDir, relativePath)
                else:
                    outputDir = outputBaseDir

                if not os.path.exists(outputDir):
                    os.makedirs(outputDir, exist_ok=True)

                analyzePolarGalaxy(fitsFilePath, outputDir, ERROR_LOG_FILE, show_plot=showPlotFlag)
            except Exception as e:
                import traceback

                error_basename = os.path.basename(fitsFilePath)
                error_message = (
                    f"\n------{error_basename}------\n"
                    f"An unexpected critical error occurred:\n{e}\n"
                    f"{traceback.format_exc()}"
                    f"------------------------------------------------------------------\n\n"
                )
                print(error_message)
                with open(ERROR_LOG_FILE, 'a') as f:
                    f.write(error_message)
