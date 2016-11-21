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
		System.out.println("CountClassVal = " + countClassVal);
		System.out.println("CountAttrNonClass = " + countAttrNonClass);
		System.out.println("CountAttrVal = " + countAttrVal);
		
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
			System.out.println("auauau");
			for (int m=0;m<a.numAttr-1;m++) {
				System.out.println("---------------------------------------");
				System.out.println("Attribute= " + data1.attribute(m).name());
				for (int n=0;n<data1.attribute(m).numValues();n++) {
					for (int p=0;p<data1.classAttribute().numValues();p++) {
						String txt = String.format("%.2f", a.probAttrNonClass[m][n][p]);
						System.out.print(txt + " ");
					}
					System.out.println();
				}
				System.out.println("---------------------------------------");
			}
			
			double accuracy = 0;
			int datatrue = 0;
			int datafalse = 0;
			for (int i=0;i<data2.numInstances();i++) {
				int indexattr = -1;
				int indexclass = -1;
				double[] probs = new double[a.numClasses];	
				double maxprob = -1;
				double probability = -1;
				String datax = "";
				for (int c=0;c<data2.classAttribute().numValues();c++) {
					double probkelas = a.probClass[c];
					probability += probkelas;
					for (int j=0;j<data2.numAttributes()-1;j++) {
						datax = data2.instance(i).stringValue(data2.attribute(j));
						for (int k=0;k<data2.attribute(j).numValues() && indexattr != 0;k++) {
							if (datax.equals(data2.attribute(j).value(k))) {
								indexattr = k;
							}
						}
						probability *= a.probAttrNonClass[j][indexattr][c];
					}
					probs[c] = probability;
				}
				for (int c=0;c<probs.length;c++) {
					if (maxprob<probs[c]) {
						maxprob = probs[c];
						indexclass = c;
					}
				}
				if (data2.instance(i).stringValue(data2.classAttribute()).equals(data2.classAttribute().value(indexclass))) {
					datatrue++;
				}
				else datafalse++;
			}
			
			System.out.println("Data yang benar = " + datatrue);
			System.out.println("Data yang salah = " + datafalse);
			accuracy = (double)datatrue/((double)datatrue+(double)datafalse);
			System.out.println("Akurasi = " + accuracy);
			
		} catch (Exception e){
			System.out.println("An error occured: " + e);
		}
	}
}
