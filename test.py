from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import invert
from skimage.morphology import medial_axis, skeletonize
from analysis.skeleton import fast_skeletonize
# from branches import detect_end_points as old_end_points, detect_branch_points as old_branch_points
from analysis.hitmiss import old_get_end_points, get_image_convolve, get_branch_point_matches, old_scan_for_end, scan_for_end, scan_for_branch
from analysis.treesearch import search_image_skeleton
from analysis.membrane import skeletons_to_membranes
import timeit
import random
import tifffile
import time
import cv2

# Generate the data
#blobs = data.binary_blobs(200, blob_size_fraction=.2,
#                          volume_fraction=.5) #, seed=1)
IMAGE = '/home/smerkous/Downloads/sample2.png' #'C:\\Users\\smerk\\Downloads\\test.png'
blobs = cv2.imread(IMAGE, cv2.IMREAD_GRAYSCALE)  # 'C:\\Users\\smerk\\UW\\Najafian Lab - Lab Najafian\\Foot Process Workspace\\out_class\\08_00816.tiff')[0]
blobs = np.ascontiguousarray(blobs)
print(blobs.shape)

def random_color():
    return (int(random.uniform(10, 255)), int(random.uniform(10, 255)), int(random.uniform(20, 255)))

blob_uint = (blobs / 255).astype(np.uint8)
skeleton = fast_skeletonize(blob_uint)
print('endpoints...')
# end_points = get_end_points(skeleton)
# old_branches = old_branch_points(skeleton)
# branch_conv = get_branch_point_matches()
# print('new branch points')
# new_branches = get_image_convolve(skeleton, branch_conv)
new_branches = scan_for_branch(skeleton, row_first=False)
# print(old_branches)
# print(end_points)
# print('new', new_branches)
end_points = scan_for_end(skeleton)

print('skeleton...')
data = search_image_skeleton(skeleton, end_points, row_first=False)
# print(np.mean(timeit.repeat(repeat=3, number=100, stmt=lambda: cv2.findContours)))

#print(skeleton[1102, 0])
#data = test_search(skeleton, end_points)
nimg = np.zeros((skeleton.shape[0], skeleton.shape[1], 3), dtype=np.uint8)
membranes = skeletons_to_membranes(data)


sizes = []
diams = []
for skel in data:
    # print('diam')
    d = skel.get_diameter()
    # print(d.get_points())
    if len(d.get_points()) > 0:
        points = d.get_points()
        sizes.append(len(points))
        diams.append(points)
        print('DIAMETER distance', d.get_distance(), 'check', cv2.arcLength(points, False))
        points = np.append(points, points[::-1], axis=0)
        cv2.drawContours(nimg, [points], 0, (255, 255, 255), 3)
        # print(points)
    
    cur_blue = 0
    for seg in skel.get_segments():
        points = seg.get_points()

        # print(set([tuple(p) for p in points]).intersection(set([tuple(p) for p in points])))
        # print('seg', points)
        print('distance', seg.get_distance(), 'check', cv2.arcLength(points, False))
        points = np.append(points, points[::-1], axis=0)
        cv2.drawContours(nimg, [points], 0, random_color(), 1)
        #color = [cur_blue, 0, 20]
        #for i, (c, r) in enumerate(points):
        #    nimg[r, c] = color
        #    color[1] = 0 + int(230 * float(i / float(len(points))))
        # cur_blue += 30

    #for x, y in skel.get_branches():
    #    # cv2.circle(nimg, branch, 1, random_color(), -1)
    #    nimg[y, x] = random_color()

    for end in skel.get_end_points():
        cv2.circle(nimg, end, 3, (255, 255, 255), 1)


blob_yes = cv2.cvtColor(blob_uint * 255, cv2.COLOR_GRAY2BGR)
for ind, mem in enumerate(membranes):
    if mem:
        print('got it')
        p = mem.get_points()
        widths = mem.get_membrane_widths(blob_uint, 0.01, 3)

        if len(p) > 0:
            print('points! compare', sizes[ind], len(p))
            print('the same', np.all(diams[ind] == p))
            points = p # mem.get_points()
            # print('DIAMETER distance', d.get_distance(), 'check', cv2.arcLength(points, False))
            points = np.append(points, points[::-1], axis=0)
            cv2.drawContours(blob_yes, [points], -1, (0, 0, 255), 2)
            print('done')
        
        """
        print('widths', widths)
        for width in widths:
            print(width.get_distance())
            cv2.line(blob_yes, width.get_first_xy(), width.get_second_xy(), (255, 255, 0), 1)
        """

    else:
        print('empty')

# print(new_branches)
#for x, y in new_branches:
#    cv2.circle(nimg, (x, y), 3, random_color(), -1)

img = nimg

# img = cv2.resize(img, (600, 600))
# cv2.circle(img, (46, 53), 3, (255, 255, 255), 1)
# img[53, 46] = (255, 255, 255)
# img[52, 47] = (255, 255, 255)
cv2.imshow('orig', img)
cv2.imshow('skel',  cv2.resize(skeleton * 255, (1000, 1000), interpolation=cv2.INTER_NEAREST))
cv2.imshow('ok', cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_NEAREST)) # cv2.INTER_LANCZOS4))
cv2.imshow('blob', blob_yes)
# cv2.imshow('skel', cv2.resize(skeleton * 255, (900, 900), interpolation=cv2.INTER_NEAREST)) # cv2.INTER_NEAREST))
cv2.waitKey(0)
# skeleton_search(skeleton, detect_end_points(skeleton))
"""
skeleton_ui = skeleton.astype(np.uint8)
"""

# points = skeleton_search(skeleton, end_points)
# print(points)

# print(timeit.timeit(lambda: scan_for_end(skeleton_ui), number=3000))  # 3 seconds
# print(timeit.timeit(lambda: old_scan_for_end(skeleton, 1104, 1024), number=3000)) # 9 seconds
# print(timeit.timeit(lambda: old_end_points(skeleton), number=3000))  # 215 seconds

"""
print('test')
found = pixel_scan(skeleton, 0, 0)
found_old = np.array([(y, x) for x, y in old_end_points(skeleton)])
print(found_old)
print(found)
"""

"""
print('time diff')
opt = (blobs / 255).astype(np.bool)
print(timeit.timeit(lambda: skeletonize(opt), number=300))
print(timeit.timeit(lambda: fast_skeletonize(opt), number=300))
cv2.imshow('skel_old', skeletonize(opt).astype(np.uint8) * 255)
cv2.imshow('skel_new', fast_skeletonize(opt).astype(np.uint8) * 255)
print('compare', np.all(skeletonize(opt).astype(np.uint8) * 255 == fast_skeletonize(opt).astype(np.uint8) * 255))
cv2.waitKey(0)
"""

"""
img = (skeleton.astype(np.int32) * 255).astype(np.uint8)
for y, x in found:
    cv2.circle(img, (x, y), 1, 255, 1)
for y, x in found_old:
    cv2.circle(img, (x, y), 3, 255, 1)
found = [(x, y) for y, x in found]
found_old = [(x, y) for y, x in found_old]

# compare old method to new one
print('nf', set(list(found)) ^ set(list(found_old)))
"""
"""
# Compute the medial axis (skeleton) and the distance transform
skel, distance = medial_axis(blobs, return_distance=True)

# Compare with other skeletonization algorithms
skeleton = skeletonize(blobs)
skeleton_lee = skeletonize(blobs, method='lee')
# detect_end_points(skeleton_lee.astype(np.bool))

# Distance to the background for pixels of the skeleton
dist_on_skel = distance * skel

fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(blobs, cmap=plt.cm.gray)
ax[0].set_title('original')
ax[0].axis('off')

ax[1].imshow(skel, cmap=plt.cm.gray)
ax[1].contour(blobs, [0.5], colors='w')
ax[1].set_title('medial_axis')
ax[1].axis('off')

ax[2].imshow(skeleton, cmap=plt.cm.gray)
ax[2].set_title('skeletonize')
ax[2].axis('off')

ax[3].imshow(skeleton_lee, cmap=plt.cm.gray)
ax[3].set_title("skeletonize (Lee 94)")
ax[3].axis('off')

fig.tight_layout()
plt.show()
"""