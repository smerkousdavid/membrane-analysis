""" Simple profiling for skeleton tree analysis """
import pstats
import cProfile
import cv2
import numpy as np

from analysis.skeleton import fast_skeletonize
from analysis.hitmiss import old_get_end_points, get_image_convolve, get_branch_point_matches, old_scan_for_end, scan_for_end, scan_for_branch
from analysis.treesearch import search_image_skeleton

IMAGE = '/home/smerkous/Downloads/sample2.png' #'C:\\Users\\smerk\\Downloads\\test.png'
blobs = cv2.imread(IMAGE, cv2.IMREAD_GRAYSCALE)  # 'C:\\Users\\smerk\\UW\\Najafian Lab - Lab Najafian\\Foot Process Workspace\\out_class\\08_00816.tiff')[0]
blobs = np.ascontiguousarray(blobs)
skeleton = fast_skeletonize((blobs / 255).astype(np.uint8))
new_branches = scan_for_branch(skeleton, row_first=False)
end_points = scan_for_end(skeleton)
data = search_image_skeleton(skeleton, end_points, row_first=False)
first_data = data[0]

cProfile.runctx("first_data.get_diameter()", globals(), locals(), "Profile.prof")

# show results
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()