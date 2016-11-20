/**
 * Created by njruntuwene on 11/15/16.
 */

import weka.classifiers.*;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Hafizh Dary
 */

public class NaiveBayes extends AbstractClassifier {
    //hasil probabilitas per value dari atribut kelas
	public double[] probClass;
	
	//hasil probabilitas per value dari setiap atribut yang bergantung pada value atribut kelas
	//indeks 1: banyak atribut non kelas
	//indeks 2: banyak jenis value dari atribut non kelas (diambil max)
	//indeks 3: banyak jenis value dari atribut kelas
	public double[][][] probAttrNonClass;
	
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
	
	public NaiveBayes() {};
    
    //get nilai maksimal dari value semua atribut non kelas (untuk keperluan array)
	public int getMaxAttrValCount() {
		int n = 0;
		int countAttrNonClass = instances.numAttributes()-1;
		for (int i=0;i<countAttrNonClass;i++) {
			int countAttrVal = instances.attribute(i).numValues();
			for (int j=0;j<countAttrVal;j++) {
				int countVal = countValAppears(i,instances.attribute(i).value(j));
				if(n<countVal) {
					n = countVal;
				}
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
			for (int n=0;n<countAttrVal;n++) {
				for (int p=0;p<countClassVal;p++) {
					int a = countValAppears(m,instances.attribute(m).value(n));
					int b = countValWithCond(m,instances.attribute(m).value(n),instances.classAttribute().value(p));
					probAttrNonClass[m][n][p] = b/a;
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
}
