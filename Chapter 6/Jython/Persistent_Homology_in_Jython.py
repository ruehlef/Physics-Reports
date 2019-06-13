# import JAVAPLEX jar into Jython
import sys
#sys.path.append('/path/to/javaplex.jar')
sys.path.append('/Users/faruehle/work/Programs/javaplex/library/javaplex.jar')

import edu.stanford.math.plex4 as p4
import edu.stanford.math.plex4.api.Plex4 as plex4
import edu.stanford.math.plex4.examples
import edu.stanford.math.plex4.io

# specify some hyperparameters for the PH computation
max_dimension = 3  # number of persistent homology dimensions to compute (b_0 to b_2 here)
max_filtration_value = 1.5  # maximum value for filtration parameter \epsilon
num_divisions = 10  # number of different intervals computed

############################################################################
# Compute the PH of a cylinder embedded in R^3 full                        #
############################################################################
# Read in the set of points
hnd = open("./cylinder.txt", "r")
res = hnd.read()
hnd.close()
point_cloud = eval(res)
print("Read in " + str(len(point_cloud)) + " points")

# Create a metric space for the point cloud (we have embedded it in R^3 and will use the Euclidean Metric)
m_space = p4.metric.impl.EuclideanMetricSpace(point_cloud)
stream = plex4.createVietorisRipsStream(m_space, max_dimension, max_filtration_value, num_divisions)

# Compute PH mod some largeish prime (37 here). Since the cylinder has not torsion, that suffices
persistence = plex4.getModularSimplicialAlgorithm(max_dimension, 37)

# Compute the intervals and transform them to filtration values
filtration_value_intervals = persistence.computeIntervals(stream)

# Print the half-open intervals for which the homologies persist
print([filtration_value_intervals])
print("########################")

# Show which complexes make up which homology element at which filtration value
intervals = persistence.computeAnnotatedIntervals(stream)
print(intervals)
print("########################")

# Create the barcode plots
p4.io.BarcodeWriter.getInstance().writeToFile(filtration_value_intervals, 0, max_filtration_value, "Cylinder b_0", "cylinder_b0.png")
p4.io.BarcodeWriter.getInstance().writeToFile(filtration_value_intervals, 1, max_filtration_value, "Cylinder b_1", "cylinder_b1.png")
p4.io.BarcodeWriter.getInstance().writeToFile(filtration_value_intervals, 2, max_filtration_value, "Cylinder b_2", "cylinder_b2.png")


############################################################################
# Compute the PH of a cylinder embedded in R^3 using lazy witness          #
############################################################################

# Set number of landmark points to use in the lazy witness complex
num_landmark_points = 100
num_divisions = 500
max_filtration_value = 2

# Compute Lazy Witness complex (with 100 randomly selected landmark points)
landmark_selector = plex4.createRandomSelector(point_cloud, num_landmark_points)
stream = plex4.createLazyWitnessStream(landmark_selector, max_dimension, max_filtration_value, num_divisions)

# Compute PH for lazy witness complex
persistence = plex4.getDefaultSimplicialAlgorithm(max_dimension)

# Compute PH for the lazy witness
filtration_value_intervals = persistence.computeIntervals(stream)

# Print the half-open intervals for which the homologies persist
print([filtration_value_intervals])
print("########################")

# Create the barcode plots
p4.io.BarcodeWriter.getInstance().writeToFile(filtration_value_intervals, 0, max_filtration_value, "LW Cylinder b_0", "lw_cylinder_b0.png")
p4.io.BarcodeWriter.getInstance().writeToFile(filtration_value_intervals, 1, max_filtration_value, "LW Cylinder b_1", "lw_cylinder_b1.png")
p4.io.BarcodeWriter.getInstance().writeToFile(filtration_value_intervals, 2, max_filtration_value, "LW Cylinder b_2", "lw_cylinder_b2.png")
