import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.*;

import static org.junit.Assert.*;

public class FFNNTest {
    private FFNN ffnn;
    private FFNN classifier;
    private Instances data;

    @Before
    public void setUp() throws Exception {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

        String filename = "/Users/njruntuwene/Desktop/iris.arff";
        DataSource source = new DataSource(filename);
        data = source.getDataSet();

        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);

        classifier = new FFNN();
        classifier.buildClassifier(data);

        double[][] temp = classifier.getInputToHiddenWeight();

        for (int i = 0; i < temp.length; i++) {
            for (int j = 0; j < temp[i].length; j++) {
                System.out.print(temp[i][j] + " ");
            }
            System.out.println();
        }

        temp = classifier.getHiddenToOutputWeight();

        for (int i = 0; i < temp.length; i++) {
            for (int j = 0; j < temp[i].length; j++) {
                System.out.print(temp[i][j] + " ");
            }
            System.out.println();
        }

        ffnn = new FFNN();
    }

//    @After
//    public void tearDown() throws Exception {
//
//    }
//
//    @Test
//    public void testSetInputLayer() throws Exception {
//
//    }
//
//    @Test
//    public void testSigmoid() throws Exception {
//
//    }
//
//    @Test
//    public void testCalculateHiddenLayer() throws Exception {
//
//    }
//
//    @Test
//    public void testCalculateOutputLayer() throws Exception {
//
//    }
//
//    @Test
//    public void testBuildClassifier() throws Exception {
//        classifier.buildClassifier(data);
//    }
//
//    @Test
//    public void testCalculateCumulativeError() throws Exception {
//
//    }
//
//    @Test
//    public void testCalculateDataError() throws Exception {
//
//    }
//
//    @Test
//    public void testBackPropagation() throws Exception {
//
//    }

    @Test
    public void testRun() throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(classifier,data);
        System.out.println(eval.toSummaryString());
    }
}