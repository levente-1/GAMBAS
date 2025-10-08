import torch
import numpy as np

def generate_slicewise_spiral_indices(H, W, D):
    # Create an empty list to store the indices in spiral order
    indices = []
    
    left, right, top, bottom = 0, W - 1, 0, H - 1

    while left <= right and top <= bottom:
        # Traverse from left to right
        for i in range(left, right + 1):
            indices.append(top * W + i)
        top += 1

        # Traverse downwards
        for i in range(top, bottom + 1):
            indices.append(i * W + right)
        right -= 1

        if top <= bottom:
            # Traverse from right to left
            for i in range(right, left - 1, -1):
                indices.append(bottom * W + i)
            bottom -= 1

        if left <= right:
            # Traverse upwards
            for i in range(bottom, top - 1, -1):
                indices.append(i * W + left)
            left += 1
    
    indices_slice1 = indices[:]

    for i in range(D-1):
        if i%2 == 0:
            indicesNew = [x+(H*W*(i+1)) for x in list(reversed(indices_slice1))]
        else:
            indicesNew = [x+(H*W*(i+1)) for x in indices_slice1]

        indices += indicesNew
    
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0).unsqueeze(0)


def gilbert2d(width, height):
    """
    Generalized Hilbert ('gilbert') space-filling curve for arbitrary-sized
    2D rectangular grids. Generates discrete 2D coordinates to fill a rectangle
    of size (width x height).
    """

    if width >= height:
        yield from generate2d(0, 0, width, 0, 0, height)
    else:
        yield from generate2d(0, 0, 0, height, width, 0)


def sgn(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


def generate2d(x, y, ax, ay, bx, by):

    w = abs(ax + ay)
    h = abs(bx + by)

    (dax, day) = (sgn(ax), sgn(ay)) # unit major direction
    (dbx, dby) = (sgn(bx), sgn(by)) # unit orthogonal direction

    if h == 1:
        # trivial row fill
        for i in range(0, w):
            yield(x, y)
            (x, y) = (x + dax, y + day)
        return

    if w == 1:
        # trivial column fill
        for i in range(0, h):
            yield(x, y)
            (x, y) = (x + dbx, y + dby)
        return

    (ax2, ay2) = (ax//2, ay//2)
    (bx2, by2) = (bx//2, by//2)

    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)

    if 2*w > 3*h:
        if (w2 % 2) and (w > 2):
            # prefer even steps
            (ax2, ay2) = (ax2 + dax, ay2 + day)

        # long case: split in two parts only
        yield from generate2d(x, y, ax2, ay2, bx, by)
        yield from generate2d(x+ax2, y+ay2, ax-ax2, ay-ay2, bx, by)

    else:
        if (h2 % 2) and (h > 2):
            # prefer even steps
            (bx2, by2) = (bx2 + dbx, by2 + dby)

        # standard case: one step up, one long horizontal, one step down
        yield from generate2d(x, y, bx2, by2, ax2, ay2)
        yield from generate2d(x+bx2, y+by2, ax, ay, bx-bx2, by-by2)
        yield from generate2d(x+(ax-dax)+(bx2-dbx), y+(ay-day)+(by2-dby),
                              -bx2, -by2, -(ax-ax2), -(ay-ay2))


def generate_slicewise_hilbert_indices(H, W, D, generator):
    indices = []
    # Origin is bottom left corner of tensor
    origin = (W*H - H)
    currentPos = origin
    # Iterate through coordinates in the generator
    for i in generator:
        # If the current position is the origin, add it to the indices
        if i == (0, 0):
            indices.append(origin)
            currentXY = i
        # Moving right in x direction
        elif currentXY[0]+1 == i[0]:
            currentPos += 1
            indices.append(currentPos)
            currentXY = i
        # Moving left in x direction
        elif currentXY[0]-1 == i[0]:
            currentPos -= 1
            indices.append(currentPos)
            currentXY = i
        # Moving up in y direction
        elif currentXY[1]+1 == i[1]:
            currentPos -= W
            indices.append(currentPos)
            currentXY = i
        # Moving down in y direction
        elif currentXY[1]-1 == i[1]:
            currentPos += W
            indices.append(currentPos)
            currentXY = i

    indices_slice1 = indices[:]

    for i in range(D-1):
        if i%2 == 0:
            indicesNew = [x+(H*W*(i+1)) for x in list(reversed(indices_slice1))]
        else:
            indicesNew = [x+(H*W*(i+1)) for x in indices_slice1]
        indices += indicesNew

    return torch.tensor(indices, dtype=torch.long).unsqueeze(0).unsqueeze(0)


def gilbert3d(width, height, depth):
    """
    Generalized Hilbert ('Gilbert') space-filling curve for arbitrary-sized
    3D rectangular grids. Generates discrete 3D coordinates to fill a cuboid
    of size (width x height x depth). Even sizes are recommended in 3D.
    """

    if width >= height and width >= depth:
       yield from generate3d(0, 0, 0,
                             width, 0, 0,
                             0, height, 0,
                             0, 0, depth)

    elif height >= width and height >= depth:
       yield from generate3d(0, 0, 0,
                             0, height, 0,
                             width, 0, 0,
                             0, 0, depth)

    else: # depth >= width and depth >= height
       yield from generate3d(0, 0, 0,
                             0, 0, depth,
                             width, 0, 0,
                             0, height, 0)


def generate3d(x, y, z,
               ax, ay, az,
               bx, by, bz,
               cx, cy, cz):

    w = abs(ax + ay + az)
    h = abs(bx + by + bz)
    d = abs(cx + cy + cz)

    (dax, day, daz) = (sgn(ax), sgn(ay), sgn(az)) # unit major direction ("right")
    (dbx, dby, dbz) = (sgn(bx), sgn(by), sgn(bz)) # unit ortho direction ("forward")
    (dcx, dcy, dcz) = (sgn(cx), sgn(cy), sgn(cz)) # unit ortho direction ("up")

    # trivial row/column fills
    if h == 1 and d == 1:
        for i in range(0, w):
            yield(x, y, z)
            (x, y, z) = (x + dax, y + day, z + daz)
        return

    if w == 1 and d == 1:
        for i in range(0, h):
            yield(x, y, z)
            (x, y, z) = (x + dbx, y + dby, z + dbz)
        return

    if w == 1 and h == 1:
        for i in range(0, d):
            yield(x, y, z)
            (x, y, z) = (x + dcx, y + dcy, z + dcz)
        return

    (ax2, ay2, az2) = (ax//2, ay//2, az//2)
    (bx2, by2, bz2) = (bx//2, by//2, bz//2)
    (cx2, cy2, cz2) = (cx//2, cy//2, cz//2)

    w2 = abs(ax2 + ay2 + az2)
    h2 = abs(bx2 + by2 + bz2)
    d2 = abs(cx2 + cy2 + cz2)

    # prefer even steps
    if (w2 % 2) and (w > 2):
       (ax2, ay2, az2) = (ax2 + dax, ay2 + day, az2 + daz)

    if (h2 % 2) and (h > 2):
       (bx2, by2, bz2) = (bx2 + dbx, by2 + dby, bz2 + dbz)

    if (d2 % 2) and (d > 2):
       (cx2, cy2, cz2) = (cx2 + dcx, cy2 + dcy, cz2 + dcz)

    # wide case, split in w only
    if (2*w > 3*h) and (2*w > 3*d):
       yield from generate3d(x, y, z,
                             ax2, ay2, az2,
                             bx, by, bz,
                             cx, cy, cz)

       yield from generate3d(x+ax2, y+ay2, z+az2,
                             ax-ax2, ay-ay2, az-az2,
                             bx, by, bz,
                             cx, cy, cz)

    # do not split in d
    elif 3*h > 4*d:
       yield from generate3d(x, y, z,
                             bx2, by2, bz2,
                             cx, cy, cz,
                             ax2, ay2, az2)

       yield from generate3d(x+bx2, y+by2, z+bz2,
                             ax, ay, az,
                             bx-bx2, by-by2, bz-bz2,
                             cx, cy, cz)

       yield from generate3d(x+(ax-dax)+(bx2-dbx),
                             y+(ay-day)+(by2-dby),
                             z+(az-daz)+(bz2-dbz),
                             -bx2, -by2, -bz2,
                             cx, cy, cz,
                             -(ax-ax2), -(ay-ay2), -(az-az2))

    # do not split in h
    elif 3*d > 4*h:
       yield from generate3d(x, y, z,
                             cx2, cy2, cz2,
                             ax2, ay2, az2,
                             bx, by, bz)

       yield from generate3d(x+cx2, y+cy2, z+cz2,
                             ax, ay, az,
                             bx, by, bz,
                             cx-cx2, cy-cy2, cz-cz2)

       yield from generate3d(x+(ax-dax)+(cx2-dcx),
                             y+(ay-day)+(cy2-dcy),
                             z+(az-daz)+(cz2-dcz),
                             -cx2, -cy2, -cz2,
                             -(ax-ax2), -(ay-ay2), -(az-az2),
                             bx, by, bz)

    # regular case, split in all w/h/d
    else:
       yield from generate3d(x, y, z,
                             bx2, by2, bz2,
                             cx2, cy2, cz2,
                             ax2, ay2, az2)

       yield from generate3d(x+bx2, y+by2, z+bz2,
                             cx, cy, cz,
                             ax2, ay2, az2,
                             bx-bx2, by-by2, bz-bz2)

       yield from generate3d(x+(bx2-dbx)+(cx-dcx),
                             y+(by2-dby)+(cy-dcy),
                             z+(bz2-dbz)+(cz-dcz),
                             ax, ay, az,
                             -bx2, -by2, -bz2,
                             -(cx-cx2), -(cy-cy2), -(cz-cz2))

       yield from generate3d(x+(ax-dax)+bx2+(cx-dcx),
                             y+(ay-day)+by2+(cy-dcy),
                             z+(az-daz)+bz2+(cz-dcz),
                             -cx, -cy, -cz,
                             -(ax-ax2), -(ay-ay2), -(az-az2),
                             bx-bx2, by-by2, bz-bz2)

       yield from generate3d(x+(ax-dax)+(bx2-dbx),
                             y+(ay-day)+(by2-dby),
                             z+(az-daz)+(bz2-dbz),
                             -bx2, -by2, -bz2,
                             cx2, cy2, cz2,
                             -(ax-ax2), -(ay-ay2), -(az-az2))


def generate_gilbert_indices_3D(H, W, D, generator):
    indices = []
    # Origin is bottom left corner of tensor
    origin = (W*H - H)
    currentPos = origin
    # Iterate through coordinates in the generator
    for i in generator:
        # If the current position is the origin, add it to the indices
        if i == (0, 0, 0):
            indices.append(origin)
            currentXYZ = i
        # Moving right in x direction
        elif currentXYZ[0]+1 == i[0]:
            currentPos += 1
            indices.append(currentPos)
            currentXYZ = i
        # Moving left in x direction
        elif currentXYZ[0]-1 == i[0]:
            currentPos -= 1
            indices.append(currentPos)
            currentXYZ = i
        # Moving up in y direction
        elif currentXYZ[1]+1 == i[1]:
            currentPos -= W
            indices.append(currentPos)
            currentXYZ = i
        # Moving down in y direction
        elif currentXYZ[1]-1 == i[1]:
            currentPos += W
            indices.append(currentPos)
            currentXYZ = i
        # Moving forward in z direction
        elif currentXYZ[2]+1 == i[2]:
            currentPos += H*W
            indices.append(currentPos)
            currentXYZ = i
        # Moving backward in z direction
        elif currentXYZ[2]-1 == i[2]:
            currentPos -= H*W
            indices.append(currentPos)
            currentXYZ = i
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0).unsqueeze(0)


if __name__ == "__main__":
 
    import argparse
 
    parser = argparse.ArgumentParser()
    parser.add_argument('height', type=int)
    parser.add_argument('width', type=int)
    parser.add_argument('depth', type=int)
    args = parser.parse_args()
    H, W, D = args.height, args.width, args.depth

    # Gilbert 3D version
    generator = gilbert3d(H, W, D)
    gilbert_indices = generate_gilbert_indices_3D(H, W, D, generator)
    column_vector = np.arange(H*W*D)
    matrix = np.zeros((H*H*D, H*H*D), dtype=int)
    matrix[column_vector, gilbert_indices] = 1

    np.save('/media/hdd/levibaljer/I2I_Mamba/indices/gilbert_eye.npy', matrix)
    np.save('/media/hdd/levibaljer/I2I_Mamba/indices/degilbert_eye.npy', np.transpose(matrix))

    gilbert_indices_r = gilbert_indices.flip(0)
    matrix = np.zeros((H*H*D, H*H*D), dtype=int)
    matrix[column_vector, gilbert_indices_r] = 1

    np.save('/media/hdd/levibaljer/I2I_Mamba/indices/degilbert_r_eye.npy', np.transpose(matrix))



