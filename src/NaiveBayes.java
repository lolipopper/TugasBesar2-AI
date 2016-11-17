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
	
    
    public void buildClassifier(Instances data) {

    }
    
    //get nilai maksimal dari value semua atribut non kelas (untuk keperluan array)
	public int getMaxAttrValCount(Instances train) {
		int n = 0;
		int countAttrNonClass = train.numAttributes()-1;
		for (int i=0;i<countAttrNonClass;i++) {
			int countAttrVal = train.attribute(i).numValues();
			for (int j=0;j<countAttrVal;j++) {
				int countVal = countValAppears(train,i,train.attribute(i).value(j));
				if(n<countVal) {
					n = countVal;
				}
			}
		}
		return n;
	}
	
    //get banyaknya kemunculan value pada semua instances
	public int countValAppears(Instances train, int attr, String val) {
		int res = 0;
		for (int i=0;i<train.numInstances();i++) {
			if (train.instance(i).stringValue(train.attribute(attr)).equals(val)) {
				res++;
			}
		}
		return res;
	}
	
    //set probabilitas kemunculan value atribut kelas per semua instance
    //baru sadar bisa pakai fungsi yang atas
	public void setProbPerClass(Instances train) {
		int countClassVal = train.classAttribute().numValues();
		probClass = new double[countClassVal];
		for (int n=0;n<countClassVal;n++) {
			int countPerClass = 0;
			for (int i=0;i<train.numInstances();i++) {
				if (train.instance(i).stringValue(train.classAttribute()).equals(train.classAttribute().value(n))) {
					countPerClass++;
				}
			}
			probClass[n] = (double)countPerClass/train.numInstances();
		}
	}
	
    //BELUM JADI
    //set conditional probability untuk setiap value dari setiap atribut non kelas
	public void setCondProbPerAttr(Instances train) {
		int countClassVal = train.classAttribute().numValues();
		int countAttrNonClass = train.numAttributes()-1;
		int countAttrVal = getMaxAttrValCount(train);
		
		probAttrNonClass = new double[countAttrNonClass][countAttrVal][countClassVal];
		
		//inisialisasi
		for (int i=0;i<countAttrNonClass;i++) {
			for (int j=0;j<countAttrVal;j++) {
				for (int k=0;k<countClassVal;k++) {
					probAttrNonClass[i][j][k] = -999; //value that won't be accessed, initiated by -999
				}
			}
		}
		
		//belum jadi
		/*int countAttr = 0;
		for (ArrayList arr : probAttrNonClass) {
			for (int n=0;n<countClassVal;n++) {
				int countAttrAppears = 0;
				int countPerClass = 0;
				for (int i=0;i<train.numInstances();i++) {
					if (train.instance(i).stringValue(train.attribute(countAttr)).equals(train.attribute(countAttr).value(n))) {
						countAttr++;
					}
				}
				arr.add((double)countPerClass/train.numInstances());
			}
		}*/
	}
}
