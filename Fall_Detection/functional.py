import numpy as np
import math

# --- Core Math Utilities ---

def approx_exp(x):
    """Approximation of exponential function for speed (mimicking C version)."""
    if x > 2.0 or x < -2.0:
        return 0.0
    x = 1.0 + x / 8.0
    x *= x
    x *= x
    x *= x
    return x

def clip(val, min_val, max_val):
    return max(min_val, min(max_val, val))

# --- Scalar Field Operations ---

def scalar_square_add_constant(field, x, y, width, v):
    """Adds a constant value 'v' to a square region around (x, y) with size 'width'."""
    minx = int(max(0, x - width))
    maxx = int(min(field.shape[1], x + width + 1))
    miny = int(max(0, y - width))
    maxy = int(min(field.shape[0], y + width + 1))

    if minx < maxx and miny < maxy:
        field[miny:maxy, minx:maxx] += v

def scalar_square_add_gauss(field, x, y, sigma, v, truncate=2.0):
    """Adds a Gaussian blob to the field."""
    for i in range(len(x)):
        cx, cy, cv, csigma = x[i], y[i], v[i], sigma[i]
        csigma2 = csigma * csigma
        radius = truncate * csigma
        
        minx = int(max(0, cx - radius))
        maxx = int(min(field.shape[1], cx + radius + 1))
        miny = int(max(0, cy - radius))
        maxy = int(min(field.shape[0], cy + radius + 1))

        for xx in range(minx, maxx):
            deltax2 = (xx - cx)**2
            for yy in range(miny, maxy):
                deltay2 = (yy - cy)**2
                if deltax2 + deltay2 > radius**2:
                    continue
                
                val = cv * math.exp(-0.5 * (deltax2 + deltay2) / csigma2)
                field[yy, xx] += val

def scalar_square_add_gauss_with_max(field, x, y, sigma, v, truncate=2.0, max_value=1.0):
    """Adds a Gaussian blob but clips the result to max_value."""
    for i in range(len(x)):
        cx, cy, cv, csigma = x[i], y[i], v[i], sigma[i]
        csigma2 = csigma * csigma
        radius = truncate * csigma
        
        minx = int(max(0, cx - radius))
        maxx = int(min(field.shape[1], cx + radius + 1))
        miny = int(max(0, cy - radius))
        maxy = int(min(field.shape[0], cy + radius + 1))

        for xx in range(minx, maxx):
            deltax2 = (xx - cx)**2
            for yy in range(miny, maxy):
                deltay2 = (yy - cy)**2
                if deltax2 + deltay2 > radius**2:
                    continue
                
                val = cv * math.exp(-0.5 * (deltax2 + deltay2) / csigma2)
                # Add value first
                field[yy, xx] += val
                # Then clip to max
                if field[yy, xx] > max_value:
                    field[yy, xx] = max_value

def scalar_square_max_gauss(field, x, y, sigma, v, truncate=2.0):
    """Takes the max of the field and a Gaussian blob."""
    for i in range(len(x)):
        cx, cy, cv, csigma = x[i], y[i], v[i], sigma[i]
        csigma2 = csigma * csigma
        radius = truncate * csigma

        minx = int(max(0, cx - radius))
        maxx = int(min(field.shape[1], cx + radius + 1))
        miny = int(max(0, cy - radius))
        maxy = int(min(field.shape[0], cy + radius + 1))

        for xx in range(minx, maxx):
            deltax2 = (xx - cx)**2
            for yy in range(miny, maxy):
                deltay2 = (yy - cy)**2
                val = cv * math.exp(-0.5 * (deltax2 + deltay2) / csigma2)
                if val > field[yy, xx]:
                    field[yy, xx] = val

def scalar_value(field, x, y, default=-1.0):
    """Returns the value at (x, y) using nearest neighbor."""
    ix, iy = int(round(x)), int(round(y))
    if 0 <= iy < field.shape[0] and 0 <= ix < field.shape[1]:
        return field[iy, ix]
    return default

def scalar_values(field, x, y, default=-1.0):
    """Batch version of scalar_value."""
    results = np.full(len(x), default, dtype=np.float32)
    for i in range(len(x)):
        results[i] = scalar_value(field, x[i], y[i], default)
    return results

def scalar_value_clipped(field, x, y):
    """Returns value at (x,y), clipping coordinates to field bounds."""
    ix = int(clip(round(x), 0, field.shape[1] - 1))
    iy = int(clip(round(y), 0, field.shape[0] - 1))
    return field[iy, ix]

def scalar_nonzero(field, x, y, default=0):
    """Returns value at (x,y) if valid, else default. (integer variant)"""
    ix, iy = int(round(x)), int(round(y))
    if 0 <= iy < field.shape[0] and 0 <= ix < field.shape[1]:
        return field[iy, ix]
    return default

def scalar_nonzero_clipped(field, x, y):
    """Returns value at (x,y), clipping coordinates."""
    ix = int(clip(round(x), 0, field.shape[1] - 1))
    iy = int(clip(round(y), 0, field.shape[0] - 1))
    return field[iy, ix]

def scalar_nonzero_clipped_with_reduction(field, x, y, r):
    """Returns value at (x/r, y/r), clipped."""
    if r == 0: return 0
    return scalar_nonzero_clipped(field, x / r, y / r)

# --- Advanced Utilities ---

def weiszfeld_nd(x, y, w=None, epsilon=1e-6, max_steps=20):
    """Calculates the geometric median."""
    if w is None:
        w = np.ones(x.shape[0])
    
    current_y = y.copy()
    
    for _ in range(max_steps):
        numerator = np.zeros_like(y)
        denominator = 0.0
        
        prev_y = current_y
        
        dist = np.linalg.norm(x - prev_y, axis=1)
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_dist = w / (dist + epsilon)
        
        denominator = np.sum(inv_dist)
        numerator = np.sum(x * inv_dist[:, np.newaxis], axis=0)
        
        if denominator == 0:
            break
            
        current_y = numerator / denominator
        
        if np.linalg.norm(current_y - prev_y) < epsilon:
            break
            
    return current_y, denominator

def caf_center_s(caf_field, x, y, sigma):
    """Finds center of CAF field given coordinates."""
    # Simplified logic: returns values from field where condition is met
    mask = (caf_field[1] > x - sigma) & (caf_field[1] < x + sigma) & \
           (caf_field[2] > y - sigma) & (caf_field[2] < y + sigma)
    return caf_field[:, mask]

def grow_connection_blend(caf_field, x, y, xy_scale, only_max=False):
    """Blends top two candidates."""
    sigma_filter = 2.0 * xy_scale
    sigma2 = 0.25 * xy_scale * xy_scale
    
    # Find candidates
    mask = (caf_field[1] > x - sigma_filter) & (caf_field[1] < x + sigma_filter) & \
           (caf_field[2] > y - sigma_filter) & (caf_field[2] < y + sigma_filter)
    
    candidates = caf_field[:, mask]
    if candidates.shape[1] == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # Calculate scores
    d2 = (candidates[1] - x)**2 + (candidates[2] - y)**2
    scores = np.exp(-0.5 * d2 / sigma2) * candidates[0]
    
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]
    
    best_idx = sorted_indices[0]
    score_1 = scores[best_idx]
    entry_1 = candidates[:, best_idx]
    
    if score_1 == 0.0:
         return 0.0, 0.0, 0.0, 0.0
         
    if only_max or len(sorted_indices) < 2:
        return entry_1[5+0], entry_1[5+1], entry_1[5+3], score_1

    # Blend with second best
    second_idx = sorted_indices[1]
    score_2 = scores[second_idx]
    entry_2 = candidates[:, second_idx]
    
    if score_2 < 0.01 or score_2 < 0.5 * score_1:
        return entry_1[5+0], entry_1[5+1], entry_1[5+3], score_1 * 0.5
        
    # Return blended result
    total_score = score_1 + score_2
    res_x = (score_1 * entry_1[5+0] + score_2 * entry_2[5+0]) / total_score
    res_y = (score_1 * entry_1[5+1] + score_2 * entry_2[5+1]) / total_score
    res_w = (score_1 * entry_1[5+3] + score_2 * entry_2[5+3]) / total_score
    
    return res_x, res_y, res_w, 0.5 * total_score

def paf_center_b(paf_field, x, y, sigma=1.0):
    """Finds center of PAF field (legacy B) given coordinates."""
    mask = (paf_field[1] > x - sigma) & (paf_field[1] < x + sigma) & \
           (paf_field[2] > y - sigma) & (paf_field[2] < y + sigma)
    return paf_field[:, mask]

def paf_center(paf_field, x, y, sigma):
    """Finds center of PAF field given coordinates."""
    return paf_center_b(paf_field, x, y, sigma)

def paf_mask_center(paf_field, x, y, sigma=1.0):
    """Creates a boolean mask for PAF field centers."""
    mask = (paf_field[1] > x - sigma * paf_field[3]) & \
           (paf_field[1] < x + sigma * paf_field[3]) & \
           (paf_field[2] > y - sigma * paf_field[3]) & \
           (paf_field[2] < y + sigma * paf_field[3])
    return mask