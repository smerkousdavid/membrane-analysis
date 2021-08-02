""" Runs all of the tests in the test suite """
import unittest

# import test modules
import tests.statistics as stats
import tests.point_centroids as point_centroids
import tests.fpw_measure as fpw_measure
import tests.point_measure as point_measure
import tests.connections as connections


# initialize the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()


# add tests to the test suite
# suite.addTests(loader.loadTestsFromModule(stats))
# suite.addTests(loader.loadTestsFromModule(point_measure))
# suite.addTests(loader.loadTestsFromModule(point_centroids))
# suite.addTests(loader.loadTestsFromModule(connections))
suite.addTests(loader.loadTestsFromModule(fpw_measure))

# initialize runner and run
runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)