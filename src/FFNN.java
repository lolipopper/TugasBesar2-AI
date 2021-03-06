import weka.classifiers.*;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

public class FFNN extends AbstractClassifier {
    private double[] inputLayer;
    private double[] hiddenLayer;
    private double[] outputLayer;
    private double[][] inputToHiddenWeight;
    private double[][] hiddenToOutputWeight;
    private Instances instances;
    private int numClasses;
    private int numAttributes;
    private int numHiddenNode = 0;
    private double[] targetValue;
    private double learningRate;
    private double minError;
    private boolean useNormalization = false;
    private boolean useHiddenLayer = true;
    private Normalize normalizeFilter;
    private Remove rm;

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
        if (!useHiddenLayer) {
            System.arraycopy(inputLayer,0,hiddenLayer,0,inputLayer.length);
            return;
        }

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

    public void setNumHiddenNode(int numHiddenNode) {
        this.numHiddenNode = numHiddenNode;
    }

    public void setOptions(String[] options) throws Exception {
        super.setOptions(options);
        this.useNormalization = Utils.getFlag('N', options);
        this.useHiddenLayer = !Utils.getFlag("no-hidden-layer",options);
        String n = Utils.getOption("num-hidden-node", options);
        if (!n.isEmpty()) this.setNumHiddenNode(Integer.parseInt(n));
        Utils.checkForRemainingOptions(options);
    }

    public String[] getOptions() {
        Vector options = new Vector();
        Collections.addAll(options, super.getOptions());
        if(this.useNormalization) {
            options.add("-N");
        }

        options.add("-num-hidden-node " + numHiddenNode);

        return (String[])options.toArray(new String[0]);
    }

    public void buildClassifier(Instances data) throws Exception {
        if (this.useNormalization) {
            this.normalizeFilter = new Normalize();
            this.normalizeFilter.setInputFormat(data);
            data = Filter.useFilter(data,normalizeFilter);
        }
        // for student only
        rm = new Remove();
        rm.setAttributeIndices("28");
        rm.setInputFormat(data);
        data = Filter.useFilter(data,rm);
        Filter numericFilter = new NominalToBinary();
        numericFilter.setInputFormat(data);
        data = Filter.useFilter(data,numericFilter);
        numClasses = data.numClasses();
        numAttributes = data.numAttributes();
        if (numHiddenNode == 0) numHiddenNode = numAttributes;
        System.out.println(numHiddenNode);
        inputLayer = new double[numAttributes];
        hiddenLayer = new double[numHiddenNode];
        outputLayer = new double[numClasses];
        inputToHiddenWeight = new double[numAttributes][numHiddenNode - 1];
        hiddenToOutputWeight = new double[numHiddenNode][numClasses];
        instances = new Instances(data);
        targetValue = new double[numClasses];
        minError = 20;
        learningRate = 0.3;

        randomizeWeight();

        double curError;
        int cnt = 0;
        do {
            if (cnt > 50000) {
                randomizeWeight();
                cnt = 0;
            }
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
//            System.out.println(curError);
            cnt++;
        } while (curError > minError);
        System.out.println("LIL");
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
        if (this.useNormalization) {
            this.normalizeFilter.input(instance);
            instance = this.normalizeFilter.output();
        }
        setInputLayer(instance);
        calculateHiddenLayer();
        calculateOutputLayer();

        return outputLayer;
    }

    public static void main(String[] argv) throws Exception {
        runClassifier(new FFNN(), argv);
    }
}
