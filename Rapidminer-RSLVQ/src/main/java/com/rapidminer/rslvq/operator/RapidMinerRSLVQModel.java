package com.rapidminer.rslvq.operator;

import java.io.Serializable;
import java.util.HashMap;
import java.util.logging.Level;

import com.rapidminer.example.Attribute;
import com.rapidminer.example.Example;
import com.rapidminer.example.ExampleSet;
import com.rapidminer.operator.OperatorException;
import com.rapidminer.operator.learner.PredictionModel;
import com.rapidminer.tools.LogService;

public class RapidMinerRSLVQModel<T extends Serializable> extends PredictionModel {

	private boolean debugMode = false;
	
	private static final long serialVersionUID = 1L;
	private ExampleSet prototypes;
	private double sigma;

	protected RapidMinerRSLVQModel(ExampleSet prototypes, ExampleSet trainingSet, double sigma) {
		super(prototypes);
		if(debugMode) LogService.getRoot().log(Level.INFO, "DEBUG: RapidminerRSLVQModel under construction");
		this.prototypes = prototypes;
		this.sigma = sigma;
	}
	
	public double getDistance(double[] x, double[] w) {
		double dist = 0.0;
		for(int i = 0; i < x.length; i++) {
			dist += Math.pow(x[i] - w[i], 2);
		}
		
		return Math.sqrt(dist);
	}
	
	// Führt eine Klassifizierung gemäß der Nearest Prototype Classification aus
	@Override
	public ExampleSet performPrediction(ExampleSet exampleSet, Attribute predictedLabel) throws OperatorException {
		if(debugMode) LogService.getRoot().log(Level.INFO, "DEBUG: RapidminerrslvModel predict"+predictedLabel.getName());
		
		for(Example example : exampleSet) {
			double[][] labelDistances = new double[prototypes.size()][2];
			
			double[] exampleValues = new double[example.getAttributes().size()];
		
			int j = 0;
			for(Attribute att : example.getAttributes()) {
				exampleValues[j] = example.getValue(att);
				j++;
			}
			int i = 0;
			for(Example cb : prototypes) {
				double[] cbValues = new double[cb.getAttributes().size()];
				j = 0;
				for(Attribute att : cb.getAttributes()) {
					cbValues[j] = cb.getValue(att);
					j++;
				}
				
				labelDistances[i][0] = getDistance(cbValues, exampleValues);
				labelDistances[i][1] = cb.getLabel();
				i++;
			}
			
			int minIndex = 0;
			double minDist = Double.MAX_VALUE;
			j = 0;
			for(double[] dist : labelDistances) {
				if(dist[0] < minDist) {
					minIndex = j;
					minDist = dist[0];
					}
				if(debugMode) LogService.getRoot().log(Level.INFO, "DEBUG: RapidminerrslvModel distances CB: "+ j + "distance: " +dist[0] + "cb.Size="+labelDistances.length);
				j++;
				}
			HashMap<Double, Double> maxDistancePerClass = new HashMap<Double, Double>();
			for(double[] dis : labelDistances) {
				maxDistancePerClass.put(dis[1], dis[0]);	
			}
			for(double[] dis : labelDistances) {
				if(maxDistancePerClass.get(dis[1]) > dis[0]) maxDistancePerClass.put(dis[1], dis[0]);	
			}
			// Anwenden der Gaussfunktion, um Wahrscheinlichkeitsdichte zu erhalten
			for(double key : maxDistancePerClass.keySet()) {
				example.setConfidence(predictedLabel.getMapping().mapIndex((int)key) , Math.exp( - ( Math.pow(maxDistancePerClass.get(key),2) / (2 * Math.pow(sigma, 2) ) ) ));
			}
			
			example.setValue(predictedLabel, labelDistances[minIndex][1]);
		}
		return exampleSet;
	}
	
	// Erzeugt den Text, der für das Model in der Result Sicht angezeigt wird
	@Override
	public String toString() {
		StringBuilder description = new StringBuilder();
		description.append(super.toString());
		description.append("\n");
		description.append("\n" + "Classification is based on Nearest Prototype Classification. Prototypes are created using RSLVQ Algorithm.\n \n");
		int i = 1;
		for(Example cb : prototypes){
			description.append("Prototype " + i + ":      ");
			description.append("Label = " + cb.getAttributes().getLabel().getName() + ": " +cb.getLabel() + "; ");
			for(Attribute att : prototypes.getAttributes()) {
				description.append(att.getName() + ": " + cb.getValue(att) + "; ");
			}
			description.append("\n");
			i++;
		}
		return description.toString();
	}
}
