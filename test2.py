from analysis.hitmiss import convolve_match, convolve_match_series
import numpy as np

img = np.array([
    [0, 0],
    [1, 1],
    [1, 1]
], dtype=np.bool)

match = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [0, 1, 0]
], dtype=np.uint8)


image = np.array([
#    0  1  2  3  4
    [0, 0, 0, 0, 0], # 0
    [0, 0, 0, 0, 0], # 1
    [0, 0, 0, 0, 0], # 2
    [0, 0, 0, 0, 0], # 3
    [0, 0, 0, 0, 0], # 4
    [0, 0, 0, 0, 0], # 5
    [0, 0, 0, 0, 0], # 6
    [0, 0, 0, 0, 0]  # 7
], dtype=np.uint8)



# print(convolve_match(img, match, 0, 1, 3, 2))
# get the width and height of the scanning region (we scan from outer edge to inner edge in a circular pattern)
start_row = 0
start_col = 0
rows, cols = image.shape
middle_row = int(rows / 2.0)
middle_col = int(cols / 2.0)

# determine width offset
if start_col >= middle_col:
    offset_col = cols - (start_col + 1)
else:
    offset_col = start_col
# determine height offset
if start_row >= middle_row:
    offset_row = rows - (start_row + 1)
else:
    offset_row = start_row

# let's get current step in the loop (step being the iteration of inner loop)
cur_step = 0
if offset_col >= offset_row and start_row < middle_row:  # more cols completed
    if start_col >= middle_col and offset_col == offset_row:
        direction = 1  # we're in the right corner let's switch direction
    else:
        direction = 0  # top row (for the rest)
    cur_step = offset_row
elif offset_col > offset_row and start_row >= middle_row:
    if start_col < middle_col and offset_col == offset_row:
        direction = 3  # left column let's switch direction
    else:
        direction = 2  # bottom row
    cur_step = offset_row
elif offset_row >= offset_col and start_col >= middle_col:
    if start_row >= middle_row and offset_col == offset_row:
        direction = 2  # finished right column let's move left
    else:
        direction = 1  # right col
    cur_step = offset_col
elif offset_row >= offset_col and start_col < middle_col:
    direction = 3  # left col
    cur_step = offset_col
else:
    raise Exception('odd loop combination for start position')

# set current loop width and height
width = cols - (cur_step * 2)
height = rows - (cur_step * 2)
cur_col = start_col
cur_row = start_row

# for our "starting position" we need to handle a couple special cases of iteration
scanned = []

while width >= 0 and height >= 0:
    offset_step = cur_step + 1
    for cur_col in range(cur_col, cols - offset_step):
        scanned.append((cur_col, cur_row))
    cur_col += 1
    for cur_row in range(cur_row, rows - offset_step):
        scanned.append((cur_col, cur_row))
    cur_row += 1
    for cur_col in range(cols - offset_step, cur_step, -1):
        scanned.append((cur_col, cur_row))
    cur_col -= 1
    for cur_row in range(rows - offset_step, offset_step, -1):
        scanned.append((cur_col, cur_row))
    cur_row -= 1

    # shift step
    cur_step += 1
    width = cols - (cur_step * 2)
    height = rows - (cur_step * 2)

sup_scan = []
for row in range(rows):
    for col in range(cols):
        sup_scan.append((col, row))

print(set(sup_scan) ^ set(scanned))

# keep iterating until we run out
"""
has_pixels = True
while has_pixels:
    if direction == 0:


for row in range(start_row, rows):
    for col in range(start_loc.x, cols):
        if image[row, col] == search:
            loc.x = col
            loc.y = row
            return <int> 1
"""

print('middle', (middle_row, middle_col))
print('offset', (offset_row, offset_col), 'direction', direction)
print('step', cur_step)
print(width, height)