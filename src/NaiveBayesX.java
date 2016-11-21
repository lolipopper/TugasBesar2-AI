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
	public static double[] probClass;
	
	//hasil probabilitas per value dari setiap atribut yang bergantung pada value atribut kelas
	//indeks 1: banyak atribut non kelas
	//indeks 2: banyak jenis value dari atribut non kelas (diambil max)
	//indeks 3: banyak jenis value dari atribut kelas
	public static double[][][] probAttrNonClass;
	
	//Hasil confusion matrix
	//index 1: jumlah data tiap kelas yang akan dites
	//index 2: hasil tes data.
	public static int[][] confusionmatrix;
	
	public Instances instances;
	public int numClasses;
	public int numAttr;
	
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
		int countAttrNonClass = instances.numAttributes()-1;
		for (int i=0;i<countAttrNonClass;i++) {
			int countAttrVal = instances.attribute(i).numValues();
			//System.out.println("AttrName = " + instances.attribute(i).name());
			//System.out.println("countAttrVal = " + countAttrVal);
			if(n<countAttrVal) {
				n = countAttrVal;
			}
		}
		return n;
	}
	
    //get frekuensi kemunculan value pada semua instances
	public int countValAppears(int attr, String val) {
		int res = 0;
		for (int i=0;i<instances.numInstances();i++) {
			if (instances.instance(i).stringValue(instances.attribute(attr)).equals(val)) {
				res++;
			}
		}
		return res;
	}
	
	//get frekuensi kemunculan value kelas saat kondisi value attr non kelas=val
	public int countValWithCond(int attr, String val, String clas) {
		int res = 0;
		for (int i=0;i<instances.numInstances();i++) {
			if (instances.instance(i).stringValue(instances.classAttribute()).equals(clas) && instances.instance(i).stringValue(instances.attribute(attr)).equals(val)) {
				res++;
			}
		}
		return res;
	}
	
    //set probabilitas kemunculan value atribut kelas per semua instance
    //baru sadar bisa pakai fungsi yang atas
	public void setProbPerClass() {
		int countClassVal = instances.classAttribute().numValues();
		probClass = new double[countClassVal];
		for (int n=0;n<countClassVal;n++) {
			int countPerClass = 0;
			for (int i=0;i<instances.numInstances();i++) {
				if (instances.instance(i).stringValue(instances.classAttribute()).equals(instances.classAttribute().value(n))) {
					countPerClass++;
				}
			}
			probClass[n] = (double)countPerClass/instances.numInstances();
		}
	}
	
    //BELUM DITES
    //set conditional probability untuk setiap value dari setiap atribut non kelas
	public void setCondProbPerAttr() {
		int countClassVal = instances.classAttribute().numValues();
		int countAttrNonClass = instances.numAttributes()-1;
		int countAttrVal = getMaxAttrValCount();
		
		probAttrNonClass = new double[countAttrNonClass][countAttrVal][countClassVal];
/*		System.out.println("CountClassVal = " + countClassVal);
		System.out.println("CountAttrNonClass = " + countAttrNonClass);
		System.out.println("CountAttrVal = " + countAttrVal);
*/		
		//inisialisasi
		for (int i=0;i<countAttrNonClass;i++) {
			for (int j=0;j<countAttrVal;j++) {
				for (int k=0;k<countClassVal;k++) {
					probAttrNonClass[i][j][k] = -1; //value that won't be accessed, initiated by -999
				}
			}
		}
		
		//belum dites
		for (int m=0;m<countAttrNonClass;m++) {
			for (int n=0;n<instances.attribute(m).numValues();n++) {
				for (int p=0;p<countClassVal;p++) {
					int a = countValAppears(m,instances.attribute(m).value(n));
					int b = countValWithCond(m,instances.attribute(m).value(n),instances.classAttribute().value(p));
					probAttrNonClass[m][n][p] = (double)b/(double)a;
				}
			}
		}
		
		
	}
	
	public void buildClassifier(Instances data) {
		instances = new Instances(data);
		numClasses = data.numClasses();
		numAttr = data.numAttributes();
		setProbPerClass();
		setCondProbPerAttr();
	}
	
	public void printAllprob(Instances data){
		for (int m=0;m<numAttr-1;m++) {
			System.out.println("---------------------------------------");
			System.out.println("Attribute= " + data.attribute(m).name());
			for (int n=0;n<data.attribute(m).numValues();n++) {
				for (int p=0;p<data.classAttribute().numValues();p++) {
					String txt = String.format("%.2f", probAttrNonClass[m][n][p]);
					System.out.print(txt + " ");
				}
				System.out.println();
			}
			System.out.println("---------------------------------------");
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
	
	public void Classify(Instances data){
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
		
		for (int i=0;i<data.numInstances();i++) {
			int indexattr = -1;
			int indexclass = -1;
			double[] probs = new double[numClasses];	
			double maxprob = -1;
			double probability = -1;
			String datax = "";
			for (int c=0;c<data.classAttribute().numValues();c++) {
				double probkelas = probClass[c];
				probability = probkelas;
				for (int j=0;j<data.numAttributes()-1;j++) {
					datax = data.instance(i).stringValue(data.attribute(j));
					for (int k=0;k<data.attribute(j).numValues() && indexattr != 0;k++) {
						if (datax.equals(data.attribute(j).value(k))) {
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
			for(int x = 0; x < data.classAttribute().numValues();x++){
				if(data.classAttribute().value(x) == data.instance(i).stringValue(data.classAttribute())){
					confusionmatrix[x][indexclass]++;
				}
			}
			if (data.instance(i).stringValue(data.classAttribute()).equals(data.classAttribute().value(indexclass))) {
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
			System.out.print("Masukkan nama file: ");
			String namafile = sc.nextLine();
			Instances data1 = DataSource.read("D:/Program Files/Weka-3-8/data/" + namafile);
			data1.setClassIndex(data1.numAttributes()-1);
			NaiveBayesX a = new NaiveBayesX();
			NumericToNominal filter = new NumericToNominal();
			filter.setInputFormat(data1);
			Instances output = Filter.useFilter(data1,filter);
			data1 = new Instances(output);
			Instances data2 = new Instances(data1);
			a.buildClassifier(data1);
			
			//Print all data
			//a.printAllprob(data1);
			
			//Classify Naive Bayes	
			a.Classify(data2);
			
		} catch (Exception e){
			System.out.println("An error occured: " + e);
		}
	}
}
