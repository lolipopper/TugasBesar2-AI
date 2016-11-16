/**
 * Created by njruntuwene on 11/15/16.
 */

import weka.classifiers.*;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Enumeration;

public class FFNN extends AbstractClassifier {
    private double[] inputLayer;
    private double[] hiddenLayer;
    private double[] outputLayer;
//    private int[][] inputToOutputWeight;
    private double[][] inputToHiddenWeight;
    private double[][] hiddenToOutputWeight;
    private Instances instances;
    private int numClasses;
    private int numAttributes;
    private int numHiddenNode;
    private double[] targetValue;
    private double minError;

    public void setInputLayer(Instance instance) {
        int attIndex = 0;
        for (Enumeration enu = instance.enumerateAttributes(); enu.hasMoreElements(); attIndex++) {
            Attribute attr = (Attribute) enu.nextElement();

            double val = instance.value(attr);
            inputLayer[attIndex] = val;
        }
    }

    public double sigmoid(int x) {
        return 1/(1+Math.exp(-x));
    }

    public void calculateHiddenLayer() {
        int[] sumWeight = new int[hiddenLayer.length];
        for (int i = 0; i < inputLayer.length; i++) {
            for (int j = 0; j < hiddenLayer.length; j++) {
                sumWeight[j] += inputLayer[i] * inputToHiddenWeight[i][j];
            }
        }

        for (int i = 0; i < hiddenLayer.length; i++) {
            hiddenLayer[i] = sigmoid(sumWeight[i]);
        }
    }

    public void calculateOutputLayer() {
        int[] sumWeight = new int[outputLayer.length];
        for (int i = 0; i < hiddenLayer.length; i++) {
            for (int j = 0; j < outputLayer.length; j++) {
                sumWeight[j] += hiddenLayer[i] * hiddenToOutputWeight[i][j];
            }
        }

        for (int i = 0; i < hiddenLayer.length; i++) {
            outputLayer[i] = sigmoid(sumWeight[i]);
        }
    }

    public void buildClassifier(Instances data) {
        numClasses = data.numClasses();
        numAttributes = data.numAttributes();
        numHiddenNode = numClasses + numAttributes + 1;
        inputLayer = new double[numAttributes];
        hiddenLayer = new double[numHiddenNode];
        outputLayer = new double[numClasses];
        inputToHiddenWeight = new double[numAttributes][numHiddenNode];
        hiddenToOutputWeight = new double[numHiddenNode][numClasses];
        instances = new Instances(data);
        targetValue = new double[numClasses];
        minError = 0.01;

        while (calculateCumulativeError() > minError) {
            for (int i = 0; i < instances.numInstances(); i++) {
                Instance curInstance = instances.instance(i);

                setInputLayer(curInstance);
                calculateHiddenLayer();
                calculateOutputLayer();

                backPropagation();
            }
        }
    }

    public double calculateCumulativeError() {
        return 0.0D;
    }

    public double calculateDataError() {
        return 0.0D;
    }

    public void backPropagation() {

    }
}
