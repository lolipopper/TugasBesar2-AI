import weka.classifiers.*;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Enumeration;
import java.util.Random;

public class FFNN extends AbstractClassifier {
    private double[] inputLayer;
    private double[] hiddenLayer;
    private double[] outputLayer;
    private double[][] inputToHiddenWeight;
    private double[][] hiddenToOutputWeight;
    private Instances instances;
    private int numClasses;
    private int numAttributes;
    private int numHiddenNode;
    private double[] targetValue;
    private double learningRate;
    private double minError;

    public FFNN() {}

    public double[] getHiddenLayer() {
        return hiddenLayer;
    }

    public double[][] getHiddenToOutputWeight() {
        return hiddenToOutputWeight;
    }

    public double[] getInputLayer() {
        return inputLayer;
    }

    public double[][] getInputToHiddenWeight() {
        return inputToHiddenWeight;
    }

    public Instances getInstances() {
        return instances;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public double getMinError() {
        return minError;
    }

    public int getNumAttributes() {
        return numAttributes;
    }

    public int getNumClasses() {
        return numClasses;
    }

    public int getNumHiddenNode() {
        return numHiddenNode;
    }

    public double[] getOutputLayer() {
        return outputLayer;
    }

    public double[] getTargetValue() {
        return targetValue;
    }

    public void setInputLayer(Instance instance) {
        double[] temp = new double[instance.numAttributes()];
        int attIndex = 0;
        Enumeration enu = instance.enumerateAttributes();
        while (enu.hasMoreElements()) {
            Attribute attr = (Attribute) enu.nextElement();

            if(attr.type() == 0) {
                double val = instance.value(attr);
                temp[attIndex] = val;
                attIndex++;
            }
        }

        inputLayer = new double[attIndex + 1];
        for (int i = 0; i < attIndex; i++) {
            inputLayer[i] = temp[i];
        }
        inputLayer[attIndex] = 1.0D;
    }

    public void setTargetValue(Instance instance) {
        for (int i = 0; i < targetValue.length; i++) {
            targetValue[i] = 0.0D;
        }
        targetValue[(int) instance.value(instance.classAttribute())] = 1.0D;
    }

    public void randomizeWeight() {
        Random rand = new Random();
        for (int i = 0; i < inputToHiddenWeight.length; i++) {
            for (int j = 0; j < inputToHiddenWeight[i].length; j++) {
                inputToHiddenWeight[i][j] = rand.nextDouble();
            }
        }

        for (int i = 0; i < hiddenToOutputWeight.length; i++) {
            for (int j = 0; j < hiddenToOutputWeight[i].length; j++) {
                hiddenToOutputWeight[i][j] = rand.nextDouble();
            }
        }
    }

    public double sigmoid(int x) {
        return 1/(1+Math.exp(-x));
    }

    public void calculateHiddenLayer() {
        int[] sumWeight = new int[hiddenLayer.length];
        for (int i = 0; i < inputLayer.length; i++) {
            for (int j = 0; j < inputToHiddenWeight[i].length; j++) {
                sumWeight[j] += inputLayer[i] * inputToHiddenWeight[i][j];
            }
        }

        for (int i = 0; i < hiddenLayer.length - 1; i++) {
            hiddenLayer[i] = sigmoid(sumWeight[i]);
        }
        hiddenLayer[hiddenLayer.length - 1] = 1.0D;
    }

    public void calculateOutputLayer() {
        int[] sumWeight = new int[outputLayer.length];
        for (int i = 0; i < hiddenLayer.length; i++) {
            for (int j = 0; j < outputLayer.length; j++) {
                sumWeight[j] += hiddenLayer[i] * hiddenToOutputWeight[i][j];
            }
        }

        for (int i = 0; i < outputLayer.length; i++) {
            outputLayer[i] = sigmoid(sumWeight[i]);
        }
    }

    public void buildClassifier(Instances data) {
        numClasses = data.numClasses();
        numAttributes = data.numAttributes();
        numHiddenNode = (numClasses + numAttributes)/2 + 1;
        inputLayer = new double[numAttributes];
        hiddenLayer = new double[numHiddenNode];
        outputLayer = new double[numClasses];
        inputToHiddenWeight = new double[numAttributes][numHiddenNode - 1];
        hiddenToOutputWeight = new double[numHiddenNode][numClasses];
        instances = new Instances(data);
        targetValue = new double[numClasses];
        minError = 1.1;
        learningRate = 0.5;

        randomizeWeight();

        double curError;
        do {
            for (int i = 0; i < instances.numInstances(); i++) {
                Instance curInstance = instances.instance(i);

                setTargetValue(curInstance);
                setInputLayer(curInstance);
                calculateHiddenLayer();
                calculateOutputLayer();

                backPropagation();
            }

            curError = 0.0D;
            for (int i = 0; i < instances.numInstances(); i++) {
                Instance curInstance = instances.instance(i);

                setTargetValue(curInstance);
                setInputLayer(curInstance);
                calculateHiddenLayer();
                calculateOutputLayer();

                // calculate data error
                double tempError = 0.0D;
                for (int j = 0; j < numClasses; j++) {
                    tempError += (outputLayer[j] - targetValue[j]) * (outputLayer[j] - targetValue[j]);
                }
                curError += (tempError / numClasses);
            }
            curError /= 2.0D;
        } while (curError > minError);
    }

    public void backPropagation() {
        double[] errOutput = new double[outputLayer.length];
        double[] errHidden = new double[hiddenLayer.length];
        for (int i = 0; i < outputLayer.length; i++) {
            errOutput[i] = outputLayer[i]*(1-outputLayer[i])*(targetValue[i]-outputLayer[i]);
            for(int j = 0; j < hiddenLayer.length; j++) {
                hiddenToOutputWeight[j][i] += learningRate*errOutput[i]*hiddenLayer[j];
            }
        }

        for (int i = 0; i < hiddenLayer.length - 1; i++) {
            errHidden[i] = 0;
            for (int j = 0; j < outputLayer.length; j++) {
                errHidden[i] += (errOutput[j]*hiddenToOutputWeight[i][j]);
            }
            errHidden[i] *= hiddenLayer[i]*(1-hiddenLayer[i]);
            for(int j = 0; j < inputLayer.length; j++) {
                inputToHiddenWeight[j][i] += learningRate*errHidden[i]*inputLayer[j];
            }
        }
    }

    public double[] distributionForInstance(Instance instance) throws Exception {
        setInputLayer(instance);
        calculateHiddenLayer();
        calculateOutputLayer();

        return outputLayer;
    }
}
