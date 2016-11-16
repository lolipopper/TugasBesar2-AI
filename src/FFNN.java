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
    private double learningRate;
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
        learningRate = 0.5;

        double curError;
        do {
            for (int i = 0; i < instances.numInstances(); i++) {
                Instance curInstance = instances.instance(i);

                setInputLayer(curInstance);
                calculateHiddenLayer();
                calculateOutputLayer();

                backPropagation();
            }

            curError = 0.0D;
            for (int i = 0; i < instances.numInstances(); i++) {
                Instance curInstance = instances.instance(i);

                setInputLayer(curInstance);
                calculateHiddenLayer();
                calculateOutputLayer();

                // calculate data error
                for (int j = 0; j < numClasses; j++) {
                    curError += (outputLayer[j] - targetValue[j]) * (outputLayer[j] - targetValue[j]);
                }
            }
            curError /= 2.0D;
        } while (curError > minError);
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
