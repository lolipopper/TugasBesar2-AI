/**
 * Created by njruntuwene on 11/15/16.
 */

import weka.classifiers.*;
import weka.core.Instance;
import weka.core.Instances;

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
    private double[] targetValue;
    private double learningRate;

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

    }

    public double calculateCumulativeError() {
        return 0.0D;
    }

    public double calculateDataError() {
        return 0.0D;
    }

    public void backPropagation() {
        double[] errOutput = new double[outputLayer.length];
        double[] errHidden = new double[hiddenLayer.length];
        for (int i=0; i<errOutput.length; i++) {
            errOutput[i] = outputLayer[i]*(1-outputLayer[i])*(targetValue[i]-outputLayer[i]);
            for(int j=0;j<hiddenToOutputWeight.length;j++) {
                hiddenToOutputWeight[j][i] += learningRate*errOutput[i]*hiddenLayer[j];
            }
        }
        for (int i=0; i<errHidden.length; i++) {
            errHidden[i] = 0;
            for (int j=0; j<hiddenToOutputWeight[i].length; j++){
                errHidden[i] += (errOutput[i]*hiddenToOutputWeight[i][j]);
            }
            errHidden[i] *= hiddenLayer[i]*(1-hiddenLayer[i]);
            for(int j=0; j<inputToHiddenWeight.length; j++) {
                inputToHiddenWeight[j][i] += learningRate*errHidden[i]*inputLayer[j];
            }
        }
    }
}
