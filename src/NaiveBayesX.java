/**
 * Created by njruntuwene on 11/15/16.
 */

import weka.classifiers.*;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.*;
import java.util.Scanner;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author Hafizh Dary
 */

public class NaiveBayesX extends AbstractClassifier {
    //hasil probabilitas per value dari atribut kelas
	public double[] probClass;
	
	//hasil probabilitas per value dari setiap atribut yang bergantung pada value atribut kelas
	//indeks 1: banyak atribut non kelas
	//indeks 2: banyak jenis value dari atribut non kelas (diambil max)
	//indeks 3: banyak jenis value dari atribut kelas
	public double[][][] probAttrNonClass;
	
	//Hasil confusion matrix
	//index 1: jumlah data tiap kelas yang akan dites
	//index 2: hasil tes data.
	public static int[][] confusionmatrix;
	
	public Instances instances;
	public Instances instanceForTrain;
	public Instances instanceForTest;
	public int numClasses;
	public int numAttr;
	public static String classattr;
	
	public Instances getInstances() {
		return instances;
	}
	
	public int getNumAttr() {
		return numAttr;
	}

	public int getNumClasses() {
		return numClasses;
	}
	
	public NaiveBayesX() {};
    
    //get nilai maksimal dari value semua atribut non kelas (untuk keperluan array)
	public int getMaxAttrValCount() {
		int n = 0;
		int countAttrNonClass = instanceForTrain.numAttributes();
		for (int i=0;i<countAttrNonClass;i++) {
			if (!instanceForTrain.attribute(i).name().equals(instanceForTrain.attribute(classattr).name())) {
				int countAttrVal = instanceForTrain.attribute(i).numValues();
				//System.out.println("AttrName = " + instances.attribute(i).name());
				//System.out.println("countAttrVal = " + countAttrVal);
				if(n<countAttrVal) {
					n = countAttrVal;
				}
			}
		}
		return n;
	}
	
    //get frekuensi kemunculan value pada semua instances
	public int countValAppears(int attr, String val) {
		int res = 0;
		for (int i=0;i<instanceForTrain.numInstances();i++) {
			if (instanceForTrain.instance(i).stringValue(instanceForTrain.attribute(attr)).equals(val)) {
				res++;
			}
		}
		return res;
	}
	
	//get frekuensi kemunculan value kelas saat kondisi value attr non kelas=val
	public int countValWithCond(int attr, String val, String clas) {
		int res = 0;
		for (int i=0;i<instanceForTrain.numInstances();i++) {
			if (instanceForTrain.instance(i).stringValue(instanceForTrain.attribute(classattr)).equals(clas) && instances.instance(i).stringValue(instances.attribute(attr)).equals(val)) {
				res++;
			}
		}
		return res;
	}
	
    //set probabilitas kemunculan value atribut kelas per semua instance
    //baru sadar bisa pakai fungsi yang atas
	public void setProbPerClass() {
		int countClassVal = instanceForTrain.attribute(classattr).numValues();
		probClass = new double[countClassVal];
		for (int n=0;n<countClassVal;n++) {
			int countPerClass = 0;
			for (int i=0;i<instanceForTrain.numInstances();i++) {
				if (instanceForTrain.instance(i).stringValue(instanceForTrain.attribute(classattr)).equals(instances.attribute(classattr).value(n))) {
					countPerClass++;
				}
			}
			probClass[n] = (double)countPerClass/instanceForTrain.numInstances();
		}
	}
	
    //BELUM DITES
    //set conditional probability untuk setiap value dari setiap atribut non kelas
	public void setCondProbPerAttr() {
		int countClassVal = instanceForTrain.attribute(classattr).numValues();
		int countAttrNonClass = instanceForTrain.numAttributes();
		int countAttrVal = getMaxAttrValCount();
		
		probAttrNonClass = new double[countAttrNonClass][countAttrVal][countClassVal];
/*		System.out.println("CountClassVal = " + countClassVal);
		System.out.println("CountAttrNonClass = " + countAttrNonClass);
		System.out.println("CountAttrVal = " + countAttrVal);
*/		
		//inisialisasi
		/*for (int i=0;i<countAttrNonClass;i++) {
			for (int j=0;j<countAttrVal;j++) {
				for (int k=0;k<countClassVal;k++) {
					probAttrNonClass[i][j][k] = -1; //value that won't be accessed, initiated by -999
				}
			}
		}*/
		
		//belum dites
		for (int m=0;m<countAttrNonClass;m++) {
			if (!instanceForTrain.attribute(m).name().equals(instanceForTrain.attribute(classattr).name())) {
				for (int n=0;n<instanceForTrain.attribute(m).numValues();n++) {
					for (int p=0;p<countClassVal;p++) {
						int a = countValAppears(m,instanceForTrain.attribute(m).value(n));
						int b = countValWithCond(m,instanceForTrain.attribute(m).value(n),instanceForTrain.attribute(classattr).value(p));

						probAttrNonClass[m][n][p] = (double)b/(double)a;
					}
				}
			}
		}
		
		
	}
	
	public Instances numericToNominal(Instances data) throws Exception {
		NumericToNominal filter = new NumericToNominal();
		filter.setInputFormat(data);
		Instances output = Filter.useFilter(data,filter);
		return output;
	}
	
	public void fullTraining(Instances data, Instances data2) {
		instanceForTrain = new Instances(data);
		instanceForTest = new Instances(data2);
		setProbPerClass();
		setCondProbPerAttr();
		/*
		for (int i=0;i<instanceForTest.numInstances();i++) {
			int indexattr = -1;
			int indexclass = -1;
			double[] probs = new double[numClasses];	
			double maxprob = -1;
			double probability;
			String datax;
			for (int c=0;c<instanceForTrain.attribute("class").numValues();c++) {
				double probkelas = probClass[c];
				probability = probkelas;
				for (int j=0;j<instanceForTrain.numAttributes()-1;j++) {
					datax = instanceForTest.instance(i).stringValue(data2.attribute(j));
					for (int k=0;k<instanceForTrain.attribute(j).numValues() && indexattr == -1;k++) {
						if (datax.equals(instanceForTrain.attribute(j).value(k))) {
							indexattr = k;
						}
					}
					probability *= probAttrNonClass[j][indexattr][c];
				}
				probs[c] = probability;
			}
			for (int c=0;c<probs.length;c++) {
				if (maxprob<probs[c]) {
					maxprob = probs[c];
					indexclass = c;
				}
			}
			if (instanceForTest.instance(i).stringValue(instanceForTest.attribute("class")).equals(instanceForTest.attribute("class").value(indexclass))) {
				datatrue++;
			}
			else datafalse++;
		}*/
	}
	
	public void savemodel(String namafile) throws Exception {
		weka.core.SerializationHelper.write(namafile, this);
	}
	
	//skala percentage = 0-100;
	public void splitTest(int percentage, Instances data2) {
		Instances data = new Instances(data2);
		data.randomize(new java.util.Random(0));
		int numTrain = (int) Math.round(data.numInstances()*percentage/100);
		int numTest = data.numInstances()-numTrain;
		instanceForTrain = new Instances(data,0,numTrain-1);
		instanceForTest = new Instances(data,numTrain,numTest);
		System.out.println(numTrain+" Training Instance, "+numTest+" Testing Instance");
		setProbPerClass();
		setCondProbPerAttr();
		/*
		for (int i=0;i<instanceForTest.numInstances();i++) {
			int indexattr = -1;
			int indexclass = -1;
			double[] probs = new double[numClasses];
			for (int arr=0;arr<probs.length;arr++) {
				probs[arr] = 0;
			}
			double maxprob = -1;
			double probability;
			String datax;
			for (int c=0;c<instanceForTrain.attribute("class").numValues();c++) {
				double probkelas = probClass[c];
				probability = probkelas;
				for (int j=0;j<instanceForTrain.numAttributes()-1;j++) {
					datax = instanceForTest.instance(i).stringValue(data2.attribute(j));
					for (int k=0;k<instanceForTrain.attribute(j).numValues() && indexattr == -1;k++) {
						if (datax.equals(instanceForTrain.attribute(j).value(k))) {
							indexattr = k;
						}
					}
					if(indexattr!=-1 && !Double.isNaN(probAttrNonClass[j][indexattr][c])) probability *= probAttrNonClass[j][indexattr][c];
				}
				probs[c] = probability;
			}
			for (int c=0;c<probs.length;c++) {
				if (maxprob<probs[c]) {
					maxprob = probs[c];
					indexclass = c;
				}
			}
			if (instanceForTest.instance(i).stringValue(instanceForTrain.attribute("class")).equals(instanceForTrain.attribute("class").value(indexclass))) {
				datatrue++;
			}
			else datafalse++;
		}*/
	}
	
	public void tenCrossValid(Instances data2) {
		splitTest(90,data2);
	}
	
	@Override
	public void buildClassifier(Instances data) {
		instances = new Instances(data);
		numClasses = data.attribute(classattr).numValues();
		numAttr = data.numAttributes();
	}
	
	public void printNaiveBayes() {
		for (int m=0;m<numAttr;m++) {
			if (!instanceForTrain.attribute(m).name().equals(instanceForTrain.attribute(classattr).name())) {
				System.out.println("---------------------------------------");
				System.out.println("Attribute= " + instanceForTrain.attribute(m).name());
				for (int n=0;n<instanceForTrain.attribute(m).numValues();n++) {
					for (int p=0;p<instanceForTrain.attribute(classattr).numValues();p++) {
						String txt = String.format("%.2f", probAttrNonClass[m][n][p]);
						System.out.print(txt + " ");
					}
					System.out.println();
				}
				System.out.println("---------------------------------------");
			}
		}
	}
	
	public void printconfusionmatrix(){
		System.out.println("---------------------------------------");
		System.out.println("CONFUSION MATRIX");
		for(int i = 0; i < confusionmatrix[0].length; i++){
			for(int j = 0; j < confusionmatrix[0].length; j++){
				System.out.print(confusionmatrix[i][j] + " ");
			}
			System.out.println();
		}
		System.out.println("---------------------------------------");
	}
	
	public void Classify(){
		double accuracy = 0;
		int datatrue = 0;
		int datafalse = 0;
		//Initialize Confusion Matrix;
		confusionmatrix = new int[numClasses][numClasses];
		for(int a = 0; a < numClasses; a++){
			for(int b = 0; b < numClasses; b++){
				confusionmatrix[a][b] = 0;
			}
		}
		
		for (int i=0;i<instanceForTest.numInstances();i++) {
			//System.out.println(i);
			int indexattr = -1;
			int indexclass = -1;
			double[] probs = new double[numClasses];	
			double maxprob = -1;
			double probability = -1;
			String datax = "";
			for (int c=0;c<instanceForTrain.attribute(classattr).numValues();c++) {
				double probkelas = probClass[c];
				probability = probkelas;
				for (int j=0;j<instanceForTrain.numAttributes();j++) {
					if (!instanceForTrain.attribute(j).name().equals(instanceForTrain.attribute(classattr).name())) {
						datax = instanceForTest.instance(i).stringValue(instanceForTest.attribute(j));
						for (int k=0;k<instanceForTrain.attribute(j).numValues() && indexattr != 0;k++) {
							if (datax.equals(instanceForTrain.attribute(j).value(k))) {
								indexattr = k;
							}
						}
						if(indexattr!=-1 && !Double.isNaN(probAttrNonClass[j][indexattr][c])) probability *= probAttrNonClass[j][indexattr][c];
					}
				}
				probs[c] = probability;
			}
			for (int c=0;c<probs.length;c++) {
				if (maxprob<probs[c]) {
					maxprob = probs[c];
					indexclass = c;
				}
			}
			for(int x = 0; x < instanceForTest.attribute(classattr).numValues();x++){
				if(instanceForTest.attribute(classattr).value(x) == instanceForTest.instance(i).stringValue(instanceForTest.attribute(classattr))){
					confusionmatrix[x][indexclass]++;
				}
			}
			if (instanceForTest.instance(i).stringValue(instanceForTest.attribute(classattr)).equals(instanceForTest.attribute(classattr).value(indexclass))) {
				datatrue++;
			}
			else datafalse++;
		}
		
		System.out.println("Data yang benar = " + datatrue);
		System.out.println("Data yang salah = " + datafalse);
		accuracy = (double)datatrue/((double)datatrue+(double)datafalse);
		String ACC = String.format("%.2f", accuracy * 100);
		System.out.println("Akurasi = " + ACC + "%");
		printconfusionmatrix();
	}
	
	public static void main(String[] args){
		try {
			Scanner sc = new Scanner(System.in);
			System.out.print("Masukkan nama file training: ");
			String namafile = sc.nextLine();
			//Instances data1 = DataSource.read("C:/Program Files/Weka-3-8/data/" + namafile);
			Instances data1 = DataSource.read("D:/Users/Richard/Desktop/TugasBesar2-AI/test/TestTubesAI2/" + namafile);
			
			//data1.setClassIndex(data1.numAttributes()-1);
			NaiveBayesX a = new NaiveBayesX();
			Instances output = a.numericToNominal(data1);
			data1 = new Instances(output);
			System.out.print("Masukkan nama file testing: ");
			namafile = sc.nextLine();
			Instances data2 = DataSource.read("D:/Users/Richard/Desktop/TugasBesar2-AI/test/TestTubesAI2/" + namafile);
			Instances output2 = a.numericToNominal(data2);
			data2 = new Instances(output2);
			classattr = "class";
			namafile = "mush.model";
			
			System.out.println("Pilihan:");
			System.out.println("1. WALC");
			System.out.println("2. DALC");
			int pil = sc.nextInt();
			if(pil == 1){
				classattr="Walc";
				namafile="Walc.model";
				/*
				Remove remove = new Remove();
				remove.setAttributeIndices("26");
				remove.setInputFormat(data1);
				output = Filter.useFilter(data1, remove);
				data1 = new Instances(output);
				Remove remove1 = new Remove();
				remove1.setAttributeIndices("26");
				remove1.setInputFormat(data2);
				output2 = Filter.useFilter(data2, remove1);
				data2 = new Instances(output2);*/
			} else if(pil == 2){
				classattr="Dalc";
				namafile="Dalc.model";
				/*
				Remove remove = new Remove();
				remove.setAttributeIndices("27");
				remove.setInputFormat(data1);
				output = Filter.useFilter(data1, remove);
				data1 = new Instances(output);
				Remove remove1 = new Remove();
				remove1.setAttributeIndices("27");
				remove1.setInputFormat(data2);
				output2 = Filter.useFilter(data2, remove1);
				data2 = new Instances(output2);
				*/
			}
			//System.out.println("DEBUG");
			a.buildClassifier(data1);
			//System.out.println("DEBUG1");
			//pilih algoritma testing
			System.out.println("Pemilihan Algoritma Testing:");
			System.out.println("1. Full Training");
			System.out.println("2. Split Test");
			System.out.println("3. 10-fold Cross Validation");
			System.out.print("Pilihan = ");
			int z = sc.nextInt();
			if(z == 1){
				a.fullTraining(data1, data2);
			} else if (z == 2){
				System.out.print("Masukkan Persentase dalam %: ");
				int y = sc.nextInt();
				a.splitTest(y, data2);
			} else if (z == 3){
				a.tenCrossValid(data2);
			} else {
				System.out.println("Wrong Choice! Quitting..");
				System.exit(0);
			}
			
			//Print all data
			a.printNaiveBayes();
			
			//Classify Naive Bayes	
			a.Classify();
			
			a.savemodel(namafile);
		} catch (Exception e){
			System.out.println("An error occured: " + e);
		}
	}
}
