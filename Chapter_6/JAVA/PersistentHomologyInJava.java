//JAVAPLEX library functionality
import edu.stanford.math.plex4.api.Plex4;
import edu.stanford.math.plex4.homology.barcodes.BarcodeCollection;
import edu.stanford.math.plex4.homology.chain_basis.Simplex;
import edu.stanford.math.plex4.homology.interfaces.AbstractPersistenceAlgorithm;
import edu.stanford.math.plex4.metric.impl.EuclideanMetricSpace;
import edu.stanford.math.plex4.streams.impl.VietorisRipsStream;
import edu.stanford.math.plex4.streams.impl.WitnessStream;
import edu.stanford.math.plex4.metric.landmark.RandomLandmarkSelector;
import edu.stanford.math.plex4.autogen.homology.IntAbsoluteHomology;
import edu.stanford.math.plex4.io.BarcodeWriter;

// Java functionality
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.Arrays;
import java.lang.String;

public class PersistentHomologyInJava {
  public static void main(String[] args) throws IOException {
    /*********************************************************************/
    /* read in file                                                      */
    /*********************************************************************/
    String inputFileName = "./cylinder.txt";
    
    Scanner input = new Scanner (new File(inputFileName));
	// pre-read in the number of rows/columns
	int rows = 0;
	int columns = 0;
	while(input.hasNextLine())
	{
		++rows;
		Scanner colReader = new Scanner(input.nextLine());
		if(rows == 1) {
		  while(colReader.hasNext())
		  {
			if(colReader.hasNextDouble()) ++columns;
			colReader.next();
		  }
		}
	}
	double[][] pointCloud = new double[rows][columns];
	input.close();

	// read in the data
	input = new Scanner(new File(inputFileName));
	for(int i = 0; i < rows; ++i)
	{
		for(int j = 0; j < columns; ++j)
		{
			if(input.hasNext())
			{
				pointCloud[i][j] = input.nextDouble();
			}
		}
	}
	input.close();
	System.out.println("Read in " + rows + " points.");
	
	
    /*********************************************************************/
    /* Compute the PH of a cylinder embedded in R^3 full                 */
    /*********************************************************************/
    
	// specify some hyperparameters for the PH computation
	int maxDimension = 3;  //number of persistent homology dimensions to compute (b_0 to b_2 here)
	double maxFiltrationValue = 1.5;  //maximum value for filtration parameter \epsilon
	int numDivisions = 200;  //number of different intervals computed
    
    // Create a metric space for the point cloud (we have embedded it in R^3 and will use the Euclidean Metric)
	System.out.println("Constructing metric space...");
	EuclideanMetricSpace mSpace = new EuclideanMetricSpace(pointCloud);
	VietorisRipsStream stream = Plex4.createVietorisRipsStream(mSpace, maxDimension, maxFiltrationValue, numDivisions);
	
	// Compute PH mod some largeish prime (37 here). Since the cylinder has not torsion, that suffices
	AbstractPersistenceAlgorithm<Simplex> persistence = Plex4.getModularSimplicialAlgorithm(maxDimension, 37);
	
	// Compute the intervals and transform them to filtration values
	BarcodeCollection<Double> filtrationValueIntervals = persistence.computeIntervals(stream);

	// Print the half-open intervals for which the homologies persist
	System.out.println(filtrationValueIntervals);
	System.out.println("########################");

	// Create the barcode plots
	BarcodeWriter.getInstance().writeToFile(filtrationValueIntervals, 0, maxFiltrationValue, "Cylinder b_0", "cylinder_b0.png");
	BarcodeWriter.getInstance().writeToFile(filtrationValueIntervals, 1, maxFiltrationValue, "Cylinder b_1", "cylinder_b1.png");
	BarcodeWriter.getInstance().writeToFile(filtrationValueIntervals, 2, maxFiltrationValue, "Cylinder b_2", "cylinder_b2.png");

	
    /*********************************************************************/
    /* Compute the PH of a cylinder embedded in R^3 using lazy witness   */
    /*********************************************************************/

	// Set number of landmark points to use in the lazy witness complex
	int numLandmarkPoints = 100;
	numDivisions = 500;
	maxFiltrationValue = 2;

	// Compute Lazy Witness complex (with 100 randomly selected landmark points)
    System.out.println("Constructing complex...");
	RandomLandmarkSelector<double[]> landmarkSelector = Plex4.createRandomSelector(mSpace, numLandmarkPoints);
    WitnessStream<double[]> wStream = Plex4.createWitnessStream(landmarkSelector, maxDimension, maxFiltrationValue, numDivisions);
	wStream.finalizeStream();

    // Compute PH for lazy witness complex
	System.out.println("Computing barcodes...");
	persistence = Plex4.getModularSimplicialAlgorithm(maxDimension, 2);
	filtrationValueIntervals = persistence.computeIntervals(wStream);

	// Print the half-open intervals for which the homologies persist
	System.out.println(filtrationValueIntervals);
	System.out.println("########################");

	// Create the barcode plots
	System.out.println("Plotting barcodes...");
	BarcodeWriter.getInstance().writeToFile(filtrationValueIntervals, 0, maxFiltrationValue, "LW Cylinder b_0", "lw_cylinder_b0.png");
	BarcodeWriter.getInstance().writeToFile(filtrationValueIntervals, 1, maxFiltrationValue, "LW Cylinder b_1", "lw_cylinder_b1.png");
	BarcodeWriter.getInstance().writeToFile(filtrationValueIntervals, 2, maxFiltrationValue, "LW Cylinder b_2", "lw_cylinder_b2.png");
	System.out.println("Done!");

  }
}
