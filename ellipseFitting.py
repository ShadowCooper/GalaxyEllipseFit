import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import os
import json
from tqdm import tqdm
import argparse
import multiprocessing as mp
from itertools import repeat

from astropy.io import fits
from astropy.visualization import ImageNormalize, HistEqStretch
from skimage.measure import find_contours
from scipy.optimize import basinhopping, minimize, differential_evolution
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist
from shapely.geometry import Polygon, Point
from shapely.affinity import scale, rotate, translate

# --- Constants ---
M0 = 22.5
PIXEL_SCALE = 0.262
ERROR_LOG_FILE = 'ellipseFittingErrors.txt'


def calculateIsophoteIntensity(targetSb, m0, pixelScale):
    """Calculates the pixel intensity corresponding to a target surface brightness."""
    return (pixelScale ** 2) * 10 ** ((m0 - targetSb) / 2.5)


def smoothContour(contour, sigma=13.0):
    """Smoothes a contour's (x, y) coordinates using a Gaussian filter."""
    ySmooth = gaussian_filter1d(contour[:, 0], sigma=sigma, mode='wrap')
    xSmooth = gaussian_filter1d(contour[:, 1], sigma=sigma, mode='wrap')
    return np.column_stack((ySmooth, xSmooth))


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
            if isinstance(key, (float, int)):
                line = f"{key:.1f}: {msg}"
            else:  # For general, non-SB level errors (e.g., key='Fatal')
                line = msg
            log_content.append(line)

    # --- Print to console ---
    print("\n" + "\n".join(log_content))

    # --- Append to file ---
    # Use a double newline to separate entries from different files.
    with open(error_file, 'a') as f:
        f.write("\n".join(log_content) + "\n\n")


def fitEllipse(contour, polarBounds=False, galaxyEllipse = None, fast = False):
    """
    Fits an ellipse to a contour by minimizing the sum of the distances from each point
    on the ellipse to the closest point on the contour.
    """
    if contour.shape[0] < 20:
        return None
    if not polarBounds:
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

        xMin, xMax, yMin, yMax = np.min(contour[:, 1]), np.max(contour[:, 1]), np.min(contour[:, 0]), np.max(contour[:, 0])
        smaSmiUpperBound = 1.2 * (np.sqrt((xMax - xMin)**2 + (yMax - yMin)**2)) / 2
        bounds = [(xMin, xMax), (yMin, yMax), (0, smaSmiUpperBound), (0, smaSmiUpperBound), (0, 2 * np.pi)]
    else:
        xcGuess, ycGuess, smaGuess, smiGuess, galAngle = galaxyEllipse['xc'], galaxyEllipse['yc'], galaxyEllipse['sma'], galaxyEllipse['smi'], galaxyEllipse['angleDeg']
        initialGuess = [xcGuess, ycGuess, smaGuess, smiGuess, np.deg2rad((galAngle - 90))]

        xMin, xMax, yMin, yMax = np.min(contour[:, 1]), np.max(contour[:, 1]), np.min(contour[:, 0]), np.max(contour[:, 0])
        smaSmiUpperBound = 1.2 * (np.sqrt((xMax - xMin)**2 + (yMax - yMin)**2)) / 2
        bounds = [(xMin, xMax), (yMin, yMax), (0, smaSmiUpperBound), (0, smaSmiUpperBound), (np.deg2rad(galAngle - 110), np.deg2rad(galAngle - 70))]

    def ellipseDist(params, contour, xy=False):
        """
        Calculates the sum of bidirectional distances between an ellipse and a contour.

        This is the sum of:
        1. The sum of the distances from each ellipse point to the closest contour point.
        2. The sum of the distances from each contour point to the closest ellipse point.

        If you only return the sum of the distances from each ellipse point to the closest
        contour point, the optimizer might just make all the ellipses tiny and close to some
        random part of the given contour. Similarly, if you only return the sum of the
        distances from each contour point to the closest ellipse point, the fit will be
        considered "good" as long as the parts of the ellipse that are close to the contour
        fit the contour well in those areas.

        Returns:
            float: The total summed distance.
        """
        xc, yc, sma, smi, angleRad = params
        if sma <= 0 or smi <= 0 or smi > sma:
            return 1e99  # Return a large number for invalid parameters

        # 1. Prepare coordinate arrays
        shapelyEllipse = createShapelyEllipse({'xc': xc, 'yc': yc, 'sma': sma, 'smi': smi, 'angleDeg': np.rad2deg(angleRad)})
        ellipsePoints = np.array(shapelyEllipse.exterior.coords)

        if not xy:
            contourPoints = np.array([p[::-1] for p in contour])
        else:
            contourPoints = np.array(contour)

        if ellipsePoints.size == 0 or contourPoints.size == 0:
            return 0.0

        # 2. Compute the full pairwise distance matrix
        distMatrix = np.linalg.norm(ellipsePoints[:, np.newaxis, :] - contourPoints[np.newaxis, :, :], axis=2)

        # 3. Find the minimum distances along each axis
        ellipseToContour = np.min(distMatrix, axis=1)
        contourToEllipse = np.min(distMatrix, axis=0)

        # 4. Sum both sets of minimum distances and return the total
        distSum = np.sum(ellipseToContour) + np.sum(contourToEllipse)

        return distSum

    #if fast:
    #    result = minimize(ellipseDist, initialGuess, args=(contour,), method='Nelder-Mead',
    #                      bounds=None)  # somehow it performs better when there are no bounds? at least in some cases?
    #else:
    #    result = differential_evolution(ellipseDist, bounds = bounds, args = (contour,))

    result = minimize(ellipseDist, initialGuess, args=(contour,), method='Nelder-Mead', bounds=None)

    xc, yc, sma, smi, angleRad = result.x

    if smi > sma:
        sma, smi = smi, sma
        angleRad += np.pi / 2
    angleDeg = np.rad2deg(angleRad) % 180.0

    return {'xc': xc, 'yc': yc, 'sma': abs(sma), 'smi': abs(smi),
            'angleDeg': angleDeg, 'angleRad': angleRad, 'contour': contour,
            'fit_error': result.fun}


def simplify_contour(contour, max_dist, galaxyEllipse):
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
                          Recommended: (0.4  or 0.5) * SMI (semi-major axis) of the galaxy
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
    min_step = 1.5 * (
                max_dist / est_spacing)  # what defines a "long enough" loop? If a point is within max_dist, but is >= 1.5 times greater contour distance than the contour distance that corresponds to a straight path away from the given point, then that loop should be removed.
    # Ensure contour is valid and sufficiently long
    if contour is None or len(contour) < min_step:
        return contour

    # 2. --- PRE-PROCESSING: RE-ORDER CONTOUR ---
    # Start the contour at the point closest to the galaxy ellipse that has an angle from the horizontal +-45 degrees from the galaxy angle. This makes it unlikely that the start point is itself inside a circuitous loop.
    gal_xc, gal_yc, galAngle, galSMA, galSMI = galaxyEllipse['xc'], galaxyEllipse['yc'], galaxyEllipse[
        'angleDeg'], galaxyEllipse['sma'], galaxyEllipse['smi']
    shapelyEllipse = createShapelyEllipse(
        {'xc': gal_xc, 'yc': gal_yc, 'sma': galSMA, 'smi': galSMI, 'angleDeg': galAngle})
    angles = np.arctan2(contour[:, 0] - gal_yc, contour[:, 1] - gal_xc)
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    angles = angles * 180 / np.pi

    minAngle = (galAngle - 45) % 360
    maxAngle = (galAngle + 45) % 360
    goodPoints = [(point, idx) for idx, (point, angle) in enumerate(zip(contour, angles)) if
                  (angle - minAngle) % 360 <= (maxAngle - minAngle) % 360]
    dists = [(shapelyEllipse.distance(Point(point[::-1])), idx) for point, idx in goodPoints]
    closestIdx = min(dists, key=lambda x: x[0])[1]

    # If we were using a NumPy array, we could use np.roll, but this way we don't have to convert back and forth
    work_contour = work_contour[closestIdx:] + work_contour[:closestIdx]

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
            # This loop correctly identifies the indices on the "short path" between i and j, handling wraparound.
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


# --- Worker function for galaxy fitting ---
def process_galaxy_sb_level(sbLevel, data, imgCenterX, imgCenterY, M0, PIXEL_SCALE, fast):
    """Processes a single SB level to find a galaxy candidate."""
    intensityGal = calculateIsophoteIntensity(sbLevel, M0, PIXEL_SCALE)
    contoursGal = find_contours(data, level=intensityGal)
    centerContourGal = max([c for c in contoursGal if len(c) > 100 and
                            c[:, 1].min() < imgCenterX < c[:, 1].max() and
                            c[:, 0].min() < imgCenterY < c[:, 0].max()],
                           key=lambda c: Polygon(c).area, default=None)
    if centerContourGal is None:
        return (sbLevel, 'Could not find this galaxy contour.')

    prelimGalFit = fitEllipse(smoothContour(centerContourGal), fast = fast)
    if not prelimGalFit:
        return (sbLevel, 'Could not fit to this galaxy contour.')

    prelimGalFit['sbLevelFound'] = sbLevel
    return prelimGalFit


# Junction finding function
def getJunctions(contour, galEllipse, minAngle, maxAngle, center):
    gal_xc, gal_yc = center
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

    def pointerCalc(pointer, change, distFraction):
        while galEllipse.distance(Point(contour[int(pointer)][::-1])) > distFraction * farthestPoint[0] and min(
                (pointer - farthestPoint[1] + n) % n, (farthestPoint[1] - pointer + n) % n) < int(n / 4):
            pointer = (pointer + change) % n
        return pointer

    pointer1 = pointer2 = farthestPoint[1]
    pointer1 = pointerCalc(pointer1, 1, 0.5)
    pointer2 = pointerCalc(pointer2, -1, 0.5)

    # If any of them was not found, try again but with distFraction = 0.7
    if min((pointer1 - farthestPoint[1] + n) % n, (farthestPoint[1] - pointer1 + n) % n) >= int(n / 4) or min((pointer2 - farthestPoint[1] + n) % n, (farthestPoint[1] - pointer2 + n) % n) >= int(n / 4):
        pointer1 = pointer2 = farthestPoint[1]
        pointer1 = pointerCalc(pointer1, 1, 0.7)
        pointer2 = pointerCalc(pointer2, -1, 0.7)

    # If any of them was not found still, it failed
    if min((pointer1 - farthestPoint[1] + n) % n, (farthestPoint[1] - pointer1 + n) % n) >= int(n / 4):
        pointer1 = None
    if min((pointer2 - farthestPoint[1] + n) % n, (farthestPoint[1] - pointer2 + n) % n) >= int(n / 4):
        pointer2 = None
    return pointer1, pointer2, farthestPoint


# --- Worker function for polar fitting ---
def process_polar_sb_level(sbLevel, data, galaxyContourPoly, galaxyEllipse, imgCenterX, imgCenterY, M0, PIXEL_SCALE,
                           junctionSmoothingSigma, fast):
    """Processes a single SB level to find polar candidates."""
    intensity = calculateIsophoteIntensity(sbLevel, M0, PIXEL_SCALE)
    potentialPeanuts = [c for c in find_contours(data, level=intensity) if len(c) > 200]

    candidates_at_this_level = []
    errors_at_this_level = []

    gal_xc, gal_yc, galAngle = galaxyEllipse['xc'], galaxyEllipse['yc'], galaxyEllipse['angleDeg']
    galaxyShapelyEllipse = createShapelyEllipse(galaxyEllipse)

    for peanutContour in potentialPeanuts:
        try:
            if not Polygon(peanutContour).contains(galaxyContourPoly):
                continue
        except:
            continue

        smoothedPeanut = smoothContour(peanutContour, sigma=junctionSmoothingSigma)
        if not fast:
            smoothedPeanut = simplify_contour(smoothedPeanut, 0.5 * galaxyEllipse['smi'], galaxyEllipse)

        junction1, junction2, farthestPoint1 = getJunctions(smoothedPeanut, galaxyShapelyEllipse,
                                                            (galAngle - 135) % 360, (galAngle - 45) % 360, (gal_xc, gal_yc))
        junction3, junction4, farthestPoint2 = getJunctions(smoothedPeanut, galaxyShapelyEllipse,
                                                            (galAngle + 45) % 360, (galAngle + 135) % 360, (gal_xc, gal_yc))
        junctionIndices = [junction1, junction2, junction3, junction4]

        if any(x is None for x in junctionIndices):
            errors_at_this_level.append((sbLevel, "Could not find all 4 junction points."))
            continue

        junctionIndices = [int(junctionIndex) for junctionIndex in junctionIndices]
        junctionIndices = sorted(junctionIndices)
        smoothedOriginalIndices = [(x, idx) for idx, x in enumerate(smoothedPeanut)]

        arcs = []
        for i in range(len(junctionIndices)):
            start, end = junctionIndices[i], junctionIndices[(i + 1) % len(junctionIndices)]
            arc = smoothedOriginalIndices[start:end] if end > start else smoothedOriginalIndices[
                                                                           start:] + smoothedOriginalIndices[:end]
            arcs.append(arc)

        polarArcs, galaxyArcs = [], []
        for arc in arcs:
            indices = {idx for _, idx in arc}
            if farthestPoint1[1] in indices or farthestPoint2[1] in indices:
                polarArcs.append(np.array([pt for pt, _ in arc]))
            else:
                galaxyArcs.append(np.array([pt for pt, _ in arc]))

        polarEllipse = fitEllipse(np.vstack(polarArcs), polarBounds = True, galaxyEllipse = galaxyEllipse, fast = fast)
        if not polarEllipse:
            errors_at_this_level.append((sbLevel, 'Could not fit to this processed polar contour.'))
            continue

        if polarEllipse['sma'] > max(data.shape): continue
        if not createShapelyEllipse(polarEllipse).contains(Point(imgCenterX, imgCenterY)): continue

        angleBetween = min(abs(galaxyEllipse['angleDeg'] - polarEllipse['angleDeg']),
                           180 - abs(galaxyEllipse['angleDeg'] - polarEllipse['angleDeg']))
        if not (70 <= angleBetween <= 110):
            errors_at_this_level.append(
                (sbLevel, "Angle between galaxy and polar ellipses wasn't between 70 and 110 degrees."))
            continue

        polarEllipse['sbLevelFound'] = sbLevel
        polarEllipse['galaxyArcs'] = galaxyArcs
        polarEllipse['polarArcs'] = polarArcs
        polarEllipse['junctionIndices'] = junctionIndices
        polarEllipse['smoothedPeanut'] = smoothedPeanut
        candidates_at_this_level.append(polarEllipse)

    return candidates_at_this_level, errors_at_this_level


def analyzePolarGalaxy(fitsFilePath, outputDirectory, error_log_path, show_plot=True, useTex=True, fast = False):
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

    # For the final .jpg image with the inverted data
    invertedData = np.max(data) - data
    norm = ImageNormalize(stretch=HistEqStretch(invertedData))

    # --- 1. Fit the Main Galaxy ---
    print(f"\nSearching for best galaxy isophote.")
    goodGalaxyCandidates = []
    sb_levels_galaxy = np.arange(23.0, 25.1, 0.1)
    sb_levels_galaxy = np.array([round(x, 1) for x in sb_levels_galaxy])

    # --- MULTIPROCESSING FOR GALAXY FITTING ---
    pool_args = zip(sb_levels_galaxy, repeat(data), repeat(imgCenterX), repeat(imgCenterY), repeat(M0),
                    repeat(PIXEL_SCALE), repeat(fast))
    with mp.Pool(processes=4) as pool:
        results = pool.starmap(process_galaxy_sb_level, pool_args)

    for res in results:
        if isinstance(res, dict):
            goodGalaxyCandidates.append(res)
        elif isinstance(res, tuple):
            # This is an error tuple (sbLevel, message)
            add_error(res[0], res[1])

    if not goodGalaxyCandidates:
        # FATAL ERROR 2: No good galaxy candidates.
        add_error('Galaxy', 'No good galaxy candidate contours.')
        log_errors(debug_log, error_log_path, baseName)
        return


    # --- FINDING BEST GALAXY CANDIDATE: Weighted average: 80% lowest fit score, 20% largest area. ---
    fitScores = np.array([e['fit_error'] / len(e['contour']) / e['sma'] for e in goodGalaxyCandidates])
    areaScores = np.array([np.pi * e['sma'] * e['smi'] for e in goodGalaxyCandidates])

    minFitScore, maxFitScore = np.min(fitScores), np.max(fitScores)
    minArea, maxArea = np.min(areaScores), np.max(areaScores)

    # Normalizing to both the max and the min is better than just normalizing to the max. That way, a single value's "goodness" is determined by how many times greater it is than the min, and how many times smaller it is than the max, essentially.
    fitScoresNorm = (fitScores - minFitScore) / (maxFitScore - minFitScore)
    areaScoresNorm = (areaScores - minArea) / (maxArea - minArea)

    finalScores = 0.8 * fitScoresNorm - 0.2 * areaScoresNorm
    bestEllipseIdx = np.argmin(finalScores)

    galaxyEllipse = goodGalaxyCandidates[bestEllipseIdx]

    # Without weights; just best fit score
    # galaxyEllipse = min(goodGalaxyCandidates, key = lambda e: e['fit_error'] / len(e['contour']) / e['sma'])


    gal_xc, gal_yc, smaGal, smiGal, angleGal = galaxyEllipse['xc'], galaxyEllipse['yc'], galaxyEllipse['sma'], galaxyEllipse['smi'], galaxyEllipse['angleDeg']
    galaxyContourPoly = Polygon(galaxyEllipse['contour'])
    galaxyShapelyEllipse = createShapelyEllipse({'xc': gal_xc, 'yc': gal_yc, 'sma': smaGal, 'smi': smiGal, 'angleDeg': angleGal})

    # --- 2. Comprehensive Search for the Best Polar Structure Candidate ---
    print("Searching all isophotes for the best polar structure candidate...")
    good_candidates = []

    sb_levels_polar = np.arange(24.5, 27.0, 0.1)
    sb_levels_polar = np.array([round(x, 1) for x in sb_levels_polar])

    # --- MULTIPROCESSING FOR POLAR FITTING ---
    pool_args_polar = zip(sb_levels_polar, repeat(data), repeat(galaxyContourPoly), repeat(galaxyEllipse),
                          repeat(imgCenterX), repeat(imgCenterY), repeat(M0), repeat(PIXEL_SCALE),
                          repeat(junctionSmoothingSigma), repeat(fast))
    with mp.Pool(processes=4) as pool:
        results_polar = pool.starmap(process_polar_sb_level, pool_args_polar)

    for candidates_at_level, errors_at_level in results_polar:
        if candidates_at_level:
            good_candidates.extend(candidates_at_level)
        if errors_at_level:
            for sb, msg in errors_at_level:
                add_error(sb, msg)

    if not good_candidates:
        # FATAL ERROR 3: No candidates found after checking all isophotes.
        # This is a fatal error, so we add it to the log.
        add_error('Fatal', 'Analysis failed: After searching all isophotes, no candidate passed all filters.')
        # Now log all collected errors (fatal and non-fatal) for this file.
        log_errors(debug_log, error_log_path, baseName)
        return

    # If we reach here, the analysis was successful.
    # The baseName for output files should not have the .fits extension.
    outputBaseName = os.path.basename(fitsFilePath).replace('.fits', '')


    # --- FINDING BEST POLAR CANDIDATE: Weighted average: 80% lowest fit score, 20% largest area. Same process as with galaxy fit. ---
    fitScores = np.array([e['fit_error'] / len(e['contour']) / e['sma'] for e in good_candidates])
    areaScores = np.array([np.pi * e['sma'] * e['smi'] for e in good_candidates])

    minFitScore, maxFitScore = np.min(fitScores), np.max(fitScores)
    minArea, maxArea = np.min(areaScores), np.max(areaScores)

    fitScoresNorm = (fitScores - minFitScore) / (maxFitScore - minFitScore)
    areaScoresNorm = (areaScores - minArea) / (maxArea - minArea)

    finalScores = 0.8 * fitScoresNorm - 0.2 * areaScoresNorm
    bestEllipseIdx = np.argmin(finalScores)

    bestPolarFit = good_candidates[bestEllipseIdx]

    # No weights, just best fit score
    # bestPolarFit = min(good_candidates, key=lambda e: e['fit_error'] / len(e['contour']) / e['sma'])


    # If we didn't simplify the contour before doing the preliminary fit: simplify, and redo the polar arc process and fitting for a better final fit.
    if fast:
        finalContour = simplify_contour(bestPolarFit['smoothedPeanut'], 0.5 * galaxyEllipse['smi'], galaxyEllipse)  # Don't use bestPolarFit['contour']; that'll make it use only the polar arcs, which will break the simplification since the simplification starts from the galaxy arcs.

        junction1, junction2, farthestPoint1 = getJunctions(finalContour, galaxyShapelyEllipse,
                                                            (angleGal - 135) % 360, (angleGal - 45) % 360, (gal_xc, gal_yc))
        junction3, junction4, farthestPoint2 = getJunctions(finalContour, galaxyShapelyEllipse,
                                                            (angleGal + 45) % 360, (angleGal + 135) % 360, (gal_xc, gal_yc))
        junctionIndices = [junction1, junction2, junction3, junction4]

        if any(x is None for x in junctionIndices):
            add_error('Fatal', f'Could not find all 4 junction indices for final simplified polar contour. (SB: {bestPolarFit['sbLevelFound']})')
            log_errors(debug_log, error_log_path, baseName)
            return

        junctionIndices = [int(junctionIndex) for junctionIndex in junctionIndices]
        junctionIndices = sorted(junctionIndices)
        simplifiedOriginalIndices = [(x, idx) for idx, x in enumerate(finalContour)]

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

        polarEllipse = fitEllipse(np.vstack(polarArcs), polarBounds = True, galaxyEllipse = galaxyEllipse, fast = fast)
        if not polarEllipse:
            add_error('Fatal', 'Could not fit to the final simplified polar contour.')
            log_errors(debug_log, error_log_path, baseName)
            return

        if polarEllipse['sma'] > max(data.shape):
            add_error('Fatal', 'Semi-major axis of the final polar ellipse is larger than the largest image dimension.')
            log_errors(debug_log, error_log_path, baseName)
            return
        if not createShapelyEllipse(polarEllipse).contains(Point(imgCenterX, imgCenterY)):
            add_error('Fatal', 'Final polar ellipse did not contain the image center.')
            log_errors(debug_log, error_log_path, baseName)
            return

        angleBetween = min(abs(galaxyEllipse['angleDeg'] - polarEllipse['angleDeg']),
                           180 - abs(galaxyEllipse['angleDeg'] - polarEllipse['angleDeg']))
        if not (70 <= angleBetween <= 110):
            add_error('Fatal', 'Angle between galaxy and final polar ellipses wasn\'t between 70 and 110 degrees.')
            log_errors(debug_log, error_log_path, baseName)
            return

        newBestEllipse = fitEllipse(finalContour, polarBounds = True, galaxyEllipse = galaxyEllipse, fast = fast)
        bestPolarFit['xc'], bestPolarFit['yc'], bestPolarFit['sma'], bestPolarFit['smi'], bestPolarFit['angleDeg'] = newBestEllipse['xc'], newBestEllipse['yc'], newBestEllipse['sma'], newBestEllipse['smi'], newBestEllipse['angleDeg']
        bestPolarFit['contour'] = newBestEllipse['contour']
        bestPolarFit['galaxyArcs'], bestPolarFit['polarArcs'] = galaxyArcs, polarArcs
        bestPolarFit['junctionIndices'] = junctionIndices
        bestPolarFit['smoothedPeanut'] = finalContour


    # --- 3. Final Results and JSON Output ---
    # We already saved some galaxy results before
    anglePolar, smaPolar, smiPolar, xcPolar, ycPolar = bestPolarFit['angleDeg'], bestPolarFit['sma'], bestPolarFit['smi'], bestPolarFit['xc'], bestPolarFit['yc']
    polarIsophoteFound = bestPolarFit['sbLevelFound']
    galaxyIsophoteFound = galaxyEllipse['sbLevelFound']
    angleBetween = min(abs(angleGal - anglePolar), 180 - abs(angleGal - anglePolar))

    helpString = "Angles are in degrees, measured counter-clockwise from the positive x-axis (horizontal). " \
                 "SMA/SMI (semi-major/minor axes) are in arcseconds. " \
                 "Isophotes are in mag/arcsec^2."
    resultsDict = {
        'galaxy': {'angle': round(angleGal, 2), 'sma': round(smaGal * PIXEL_SCALE, 2),
                   'smi': round(smiGal * PIXEL_SCALE, 2), 'isophote': galaxyIsophoteFound,
                   'xc': round(gal_xc, 2),  'yc': round(gal_yc, 2)},
        'polarStructure': {'angle': round(anglePolar, 2), 'sma': round(smaPolar * PIXEL_SCALE, 2),
                           'smi': round(smiPolar * PIXEL_SCALE, 2), 'isophote': round(polarIsophoteFound, 1),
                           'xc': round(xcPolar, 2), 'yc': round(ycPolar, 2)},
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
        f"Polar Angle:  {anglePolar:.2f}°, SMA: {smaPolar:.2f} pix, SMI: {smiPolar:.2f} pix (from polar isophote at {polarIsophoteFound:.1f} mag/arcsec^2)")
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

        if useTex:
            galaxyLabel = rf'Galaxy Fit (SB = {galaxyIsophoteFound} $\frac{{\mathrm{{mag}}}}{{{{\mathrm{{arcsec}}}}^2}}$)'
            polarLabel = rf'Polar Fit (SB = {polarIsophoteFound} $\frac{{\mathrm{{mag}}}}{{{{\mathrm{{arcsec}}}}^2}}$)'
        else:
            galaxyLabel = f'Galaxy Fit (SB = {galaxyIsophoteFound} mag/arcsec^2)'
            polarLabel = f'Polar Fit (SB = {polarIsophoteFound} mag/arcsec^2)'
        eGal = Ellipse(xy=(gal_xc, gal_yc), width=2 * smaGal, height=2 * smiGal,
                       angle=angleGal, edgecolor='cyan', facecolor='none', lw=2,
                       label=galaxyLabel)
        ax.add_patch(eGal)

        ePolar = Ellipse(xy=(xcPolar, ycPolar), width=2 * smaPolar, height=2 * smiPolar,
                         angle=anglePolar, edgecolor='magenta', facecolor='none', lw=2,
                         label=polarLabel)
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
    parser.add_argument("--no-tex", action="store_true",
                        help="Stop LaTeX formatting in the final JPG plots. If you are getting PyQt5/LaTeX-related errors, this will stop them.")
    parser.add_argument("--fast", action="store_true", help="Make the script faster by only simplifying the final selected contour. May yield worse results.")
    args = parser.parse_args()

    inputPath = args.inputPath
    outputBaseDir = args.outputDirectory
    showPlotFlag = not args.no_plot
    useTexFlag = not args.no_tex
    fastFlag = args.fast

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

                if useTexFlag:
                    matplotlib.use('Qt5Agg')
                    plt.rcParams['text.usetex'] = True

                analyzePolarGalaxy(fitsFilePath, outputDir, ERROR_LOG_FILE, show_plot=showPlotFlag, useTex=useTexFlag, fast = fastFlag)
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
