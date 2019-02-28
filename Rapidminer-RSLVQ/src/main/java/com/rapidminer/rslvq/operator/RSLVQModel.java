package com.rapidminer.rslvq.operator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.logging.Level;
import com.rapidminer.example.Attribute;
import com.rapidminer.example.Attributes;
import com.rapidminer.example.Example;
import com.rapidminer.example.ExampleSet;
import com.rapidminer.example.table.AttributeFactory;
import com.rapidminer.example.utils.ExampleSetBuilder;
import com.rapidminer.example.utils.ExampleSets;
import com.rapidminer.operator.OperatorException;
import com.rapidminer.operator.learner.PredictionModel;
import com.rapidminer.tools.LogService;
import com.rapidminer.tools.RandomGenerator;

public class RSLVQModel {
	
	// Attribute für Debugging
	private boolean debugMode = false;
	int outputMinimizer = 0;
	
	//Objektvariablen
	private ExampleSet prototypes;
	private PredictionModel model;
	private ExampleSet trainingSet;
	private double sigma;
	private double learningRate;
	private double updateRate;
	private double decay_factor;
	private boolean cost_log;
	private double pj;
	private ExampleSet costProgress;
	
	//Constructor
	public RSLVQModel(ExampleSet initialPrototypes, double sigma, int ppc, double updateRate, double decay_factor, boolean cost_log) {
		this.prototypes = initialPrototypes;
		this.sigma = sigma;
		this.updateRate = updateRate;
		this.learningRate = updateRate;
		this.decay_factor = decay_factor;
		this.cost_log = cost_log;
	}
	
	// Berechnet euklidische Distanz zwischen zwei Examples und wendet Gaussfunktion an
	public double getD(Example x, Example w) {

		double[] exampleAttributes = new double[x.getAttributes().size()];
		double[] prototypeAttributes = new double[w.getAttributes().size()];
		int i = 0;
		
		for(Attribute attribute : x.getAttributes()) {
			exampleAttributes[i] = x.getValue(attribute);
			i++;
		}
		
		i = 0;
		for(Attribute attribute : w.getAttributes()) {
			prototypeAttributes[i] = w.getValue(attribute);
			i++;
		}
		
		double dist = 0.0;
		
		for(i = 0; i < exampleAttributes.length; i++) {
			dist += Math.pow(exampleAttributes[i] - prototypeAttributes[i], 2);
		}
		
		dist = Math.sqrt(dist);
		
		return  (Math.pow(dist,2) / (2 * Math.pow(sigma, 2)));
	}
	
	// Berechnet bedingte oder unbedingte Wahrscheinlichkeitsdichte
	public double getP(double exToCb, double distances) throws OperatorException {
		if(distances == 0) return 0;
		return exToCb / distances;
	}
	
	// Optimiert die Prototypen gemäß dem RSLVQ Algorithmus
	// Erzeugt ein Model, das in RapidMiner zur Klassifizierung verwendet werden kann
	public void fit(ExampleSet trainingSet, int iterations, RandomGenerator randomGenerator) throws OperatorException {
		this.trainingSet = trainingSet;
		
		// Speichert alle Attribute
		ArrayList<Attribute> attributeList = new ArrayList<>();
		Iterator<Attribute> attributes = trainingSet.getAttributes().allAttributes();
		for ( Iterator<Attribute> attIter = attributes; attIter.hasNext();){
			attributeList.add((Attribute) attIter.next());
		}
		String[] attributesStr = new String[attributeList.size()];
		int i = 0;
		for(Attribute att : attributeList) {
			attributesStr[i] = trainingSet.getAttributes().get(att.getName()).getName();
			i++;
		}
		
		//Erzeuge ExampleSet um den Kostenverlauf des Models zu dokumentieren
		Attribute costAtt = AttributeFactory.createAttribute("Cost", 3);
		Attribute epochAtt = AttributeFactory.createAttribute("Epoch", 2);
		ExampleSetBuilder builder = ExampleSets.from(costAtt, epochAtt);
		double[] costValues = new double[2];
		
		if(cost_log) LogService.getRoot().log(Level.INFO, "Initial Cost: " + getCost());
		
		// anwenden der Update Rule für jedes Example im Trainingsets
		// zufällige Reihenfolge des Trainingsets
		// Schleife in Abhängigkeit des Iterationsparameter
		for(int iter = 1; iter <= iterations; iter++) {
			i = 1;

			// UpdateRate ist die initiale LearningRate
			// learningRate wird durch decay_factor nach jeder epoche geringer
			learningRate = ( updateRate / (1 + (iter - 1) * decay_factor)) /Math.pow(sigma, 2);
			
			if(cost_log) LogService.getRoot().log(Level.INFO, "Learning Rate: " + learningRate);
			
			// order wird für den zufälligen Zugriff verwendet
			ArrayList<Integer> order = new ArrayList<>();
			for(int orderI = 0; orderI < this.trainingSet.size(); orderI++) {
				order.add(orderI);
			}
			
			
			//newOrder hält die Zugriffsschlüssel zu den Trainingsdaten in zufälliger Reihenfolge
			ArrayList<Integer> newOrder = new ArrayList<>();
			while(! (order.size() == 0)) {
				int nextI = 0;
				if(!(order.size() == 1)) {
					nextI = randomGenerator.nextInt(order.size()-1);
				}
				
				newOrder.add(order.get(nextI));
				order.remove(nextI);
			}
					
			// Schleife über die Trainingsdaten in zufälliger Reihenfolge
			int updateIndex = 1;
			for(Integer index : newOrder) {
				if(debugMode) LogService.getRoot().log(Level.INFO, "UPDATE INDEX " + updateIndex + "|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||");
				if(debugMode) RSLVQ_Operator.printExample(trainingSet.getExample(index));
				
				double[] dCbs = new double[prototypes.size()];
				double dCorrect = 0.0;
				double dAll = 0.0;
				
				// Speichert alle notwendigen Distanzmaße/Wahrscheinlichkeiten für die Prototypen
				// (dCbs für einzelne Prototypen/ dCorrect für Prototypen mit übereinstimmendem Label zum Exemple aus dem Trainingset/ dAll für die Summe der Distanzen vom Example zu allen Prototypen)
				int cbI = 0;
				for(Example prototype : prototypes) {
					double dist = getD(trainingSet.getExample(index), prototype);
					dCbs[cbI] = Math.exp(-dist);
					dAll += Math.exp(-dist);
					if(trainingSet.getExample(index).getLabel() == prototype.getLabel()) {
						dCorrect += Math.exp(-dist);
					}
					cbI++;
				}
				
				if(debugMode) LogService.getRoot().log(Level.INFO, " All Distances correct " + dCorrect + " all " + dAll + " distArr" + Arrays.toString(dCbs));
				
				// Schleife über alle Prototypen
				// Anwendung der UpdateRule
				int j = 0;
				for(Example prototype : prototypes) {
					double plx = getP(dCbs[j], dAll);
					if(this.trainingSet.getExample(index).getLabel() == prototype.getLabel()) { // if example and prototype label are compliant
						double pylx = getP(dCbs[j], dCorrect);
						if(debugMode) LogService.getRoot().log(Level.INFO, "|||||||||||||||||||||||||||||||||||||||||||  Prototyp " + j + " Update mit  EQUAL LABELS epoch " + updateIndex);
						if(debugMode) RSLVQ_Operator.printExample(prototype);
						if(debugMode) LogService.getRoot().log(Level.INFO, "DEBUG:FIT()        pylx - plx = " + pylx +"  -  " + plx + " Distance Ex to Cb " + dCbs[j] + " All Distances " + dAll + "distArr" + Arrays.toString(dCbs));
						int attCounter = 0;
						for(Attribute att : prototype.getAttributes()) {
							double diff = ((this.trainingSet.getExample(index).getValue(this.trainingSet.getExample(index).getAttributes().get(att.getName()))) - prototype.getValue(att));
							double oldValue = prototype.getValue(att);
							double newValue = (oldValue + learningRate * (pylx - plx) * diff);
							if(debugMode) LogService.getRoot().log(Level.INFO, "DEBUG: calculated val   " + newValue);
							if(debugMode) LogService.getRoot().log(Level.INFO, "DEBUG: set new Value   " + newValue);
							if(oldValue != newValue) prototype.setValue(att, newValue);
							if(debugMode) LogService.getRoot().log(Level.INFO, "DEBUG: val   " + prototype.getValue(att));
							if(debugMode && (attCounter % 1 == 0)) LogService.getRoot().log(Level.INFO, "DEBUG:FIT()        Distanz :  " + dCbs[j] + " or " + getD(this.trainingSet.getExample(index), prototype) + "   learning faktor: " + (learningRate) + " " + (pylx - plx) + "   Differenz: " + diff + " Value " + oldValue + "=>" + (oldValue + learningRate * (pylx - plx) * diff) + "delta " + (oldValue - prototype.getValue(att)));
							attCounter++;
						}
					}else {
						if(debugMode) LogService.getRoot().log(Level.INFO, "|||||||||||||||||||||||||||||||||||||||||||  Prototyp " + j + " Update  mit  UNEQUAL LABELS epoch " + updateIndex);
						if(debugMode) RSLVQ_Operator.printExample(prototype);
						if(debugMode) LogService.getRoot().log(Level.INFO, "DEBUG:FIT()    plx = " + "  -  " + plx + " Distance Ex to Cb " + dCbs[j] + " All Distances " + dAll + "distArr" + Arrays.toString(dCbs));
						int attCounter = 0;
						for(Attribute att : prototype.getAttributes()) {
							double diff = ((this.trainingSet.getExample(index).getValue(this.trainingSet.getExample(index).getAttributes().get(att.getName()))) - prototype.getValue(att));
							double oldValue = prototype.getValue(att);
							double newValue = (oldValue - learningRate * plx * diff);
							if(debugMode) LogService.getRoot().log(Level.INFO, "DEBUG: new calculatedval   " + newValue);
							if(debugMode) LogService.getRoot().log(Level.INFO, "DEBUG: set Value:   " + newValue);
							if(oldValue != newValue) prototype.setValue(att, newValue);
							if(debugMode) LogService.getRoot().log(Level.INFO, "DEBUG: newVal   " + prototype.getValue(att));
							if(debugMode && (attCounter % 1 == 0)) LogService.getRoot().log(Level.INFO, "DEBUG:FIT()        Distanz :  " + dCbs[j] + " or " + getD(this.trainingSet.getExample(index), prototype) + "   learning faktor: " + (learningRate) + " " + (- plx) + "   Differenz: " + diff  + " Value " + oldValue + "=>" + (oldValue - learningRate * plx * diff) + "delta " + (oldValue - prototype.getValue(att)));
							attCounter++;
						}
					}
					j++;
				}
				i++;
				updateIndex++;
			}
			
			// Erzeugt die Example für den Kostenverlauf des Models
			if(cost_log) {
				LogService.getRoot().log(Level.INFO, "Cost after epoch " + iter + ": " + getCost());
				costValues[0] = iter;
				costValues[1] = getCost();
				builder.addRow(costValues);
			}
			
		}
		// Erzeugt das ExampleSet für den Kostenverlauf des Models
		costProgress = builder.build();
		
		// Erzeugt das Model, für die Weiterverwendung in RapidMiner
		RapidMinerRSLVQModel<Double> testModel = new RapidMinerRSLVQModel<Double>(prototypes, trainingSet, sigma);
		model = testModel;
	}
	
		
	public PredictionModel getPredictionModel() {
		return model;
	}
	
	public ExampleSet getPrototypes() {
		return prototypes;
	}
	
	public double getCost() {
		double cost = 0.0;
		pj = 1.0/prototypes.size();
		for(Example example : trainingSet) {
			cost += Math.log(get_pxyT(example)/get_pxT(example));
		}
		return cost;
	}
	
	public double get_pxyT(Example example) {
		double pxyT = 0.0;
		for(Example prototype : prototypes) {
			if(example.getLabel() == prototype.getLabel()) {
				pxyT += (Math.exp(-getD(example, prototype)) * pj);
			}
		}
		return pxyT;
	}
	
	public double get_pxT(Example example) {
		double pxT = 0.0;
		for(Example prototype : prototypes) {
			pxT += (Math.exp(-getD(example, prototype)) * pj);
		}
		return pxT;
	}
	
	public ExampleSet getCostProgress() {
		return costProgress;
	}
	
}
