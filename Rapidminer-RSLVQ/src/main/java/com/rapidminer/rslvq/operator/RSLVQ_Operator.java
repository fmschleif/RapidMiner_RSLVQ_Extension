package com.rapidminer.rslvq.operator;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.logging.Level;

import com.rapidminer.example.Attribute;
import com.rapidminer.example.Attributes;
import com.rapidminer.example.Example;
import com.rapidminer.example.ExampleSet;
import com.rapidminer.example.table.AttributeFactory;
import com.rapidminer.operator.Operator;
import com.rapidminer.operator.OperatorCapability;
import com.rapidminer.operator.OperatorDescription;
import com.rapidminer.operator.OperatorException;
import com.rapidminer.operator.ports.InputPort;
import com.rapidminer.operator.ports.OutputPort;
import com.rapidminer.operator.ports.metadata.CapabilityPrecondition;
import com.rapidminer.operator.learner.CapabilityProvider;
import com.rapidminer.operator.learner.PredictionModel;
import com.rapidminer.operator.ports.metadata.DistanceMeasurePrecondition;
import com.rapidminer.operator.ports.metadata.ExampleSetMetaData;
import com.rapidminer.operator.ports.metadata.ExampleSetPrecondition;
import com.rapidminer.operator.ports.metadata.GeneratePredictionModelTransformationRule;
import com.rapidminer.operator.ports.metadata.MDInteger;
import com.rapidminer.operator.ports.metadata.MetaData;
import com.rapidminer.operator.ports.metadata.PassThroughRule;
import com.rapidminer.parameter.ParameterType;
import com.rapidminer.parameter.ParameterTypeBoolean;
import com.rapidminer.parameter.ParameterTypeDouble;
import com.rapidminer.parameter.ParameterTypeInt;
import com.rapidminer.parameter.UndefinedParameterError;
import com.rapidminer.example.utils.ExampleSetBuilder;
import com.rapidminer.example.utils.ExampleSets;
import com.rapidminer.tools.LogService;
import com.rapidminer.tools.RandomGenerator;
import com.rapidminer.tools.math.similarity.DistanceMeasures;
import cern.colt.Arrays;

public class RSLVQ_Operator extends Operator implements CapabilityProvider {
	
	// Attribute für Debugging
	boolean debugMode = false;
	static Attributes debugAtt;

	// InputPort
	private InputPort exampleSetInputPort = getInputPorts().createPort("example set");
	
	//OutputPorts
	private OutputPort modelOutputPort = getOutputPorts().createPort("model");
	private OutputPort prototypesOutputPort = getOutputPorts().createPort("prototypes");
	private OutputPort exampleSetOutputPort = getOutputPorts().createPort("example set");
	private OutputPort costProgressOutputPort = getOutputPorts().createPort("cost progress");
	
	//Parameters
	public static final String PARAMETER_PROTOTYPES_PER_CLASS = "prototypes per class";
	public static final String PARAMETER_ITERATIONS = "iterations";
	public static final String PARAMETER_UPDATE_RATE = "alpha";
	public static final String PARAMETER_DECAY_FACTOR = "decay";
	public static final String PARAMETER_SIGMA ="sigma";
	public static final String PARAMETER_COST_LOG = "error log";
	
	//Objektvariablen
	private int ppc;
	private int iterations;
	private double updateRate;
	private double decayFactor;
	private double sigma;
	public boolean costLog;
	
	//Constructor - wird aufgerufen, sobald der Operator auf den Process Panel gezogen wird
	public RSLVQ_Operator(OperatorDescription description) {
		super(description);
		
		//Preconditions zu den Ports hinzufügen
		exampleSetInputPort.addPrecondition(new CapabilityPrecondition(this, exampleSetInputPort));
        exampleSetInputPort.addPrecondition(new DistanceMeasurePrecondition(exampleSetInputPort, this));
        exampleSetInputPort.addPrecondition(new ExampleSetPrecondition(exampleSetInputPort, "label", 0));
        exampleSetInputPort.addPrecondition(new CapabilityPrecondition(new CapabilityProvider() {
            @Override
            public boolean supportsCapability(OperatorCapability capability) {
                switch (capability) {
                	// case BINOMINAL_ATTRIBUTES: Not possible because it only accepts values 1 and -1 (otherwise NaN)
                    case POLYNOMINAL_ATTRIBUTES:
                    case NUMERICAL_ATTRIBUTES:
                    case POLYNOMINAL_LABEL:
                    case BINOMINAL_LABEL:
                        return true;
                    default:
                        return false;
                }
            }
        }, exampleSetInputPort));
        
        //Transformationsregeln für MetaData hinzufügen
        getTransformer().addRule(new GeneratePredictionModelTransformationRule(exampleSetInputPort, modelOutputPort, RapidMinerRSLVQModel.class));        
        getTransformer().addPassThroughRule(exampleSetInputPort, exampleSetOutputPort);
        addPrototypeTransformationRule();
		
        //Standardwerte für die Operatorparameter festlegen
        ppc = 1;
		iterations = ppc * 50;
		updateRate = 0.1;
		decayFactor = 0.85;
		sigma = 0.5;
		costLog = false;
	}
	
	// Wird mit der Prozessauführung in RapidMiner aufgerufen, sobald der Operator an der Reihe ist
	// Erstellt Prototypen, führt das Training unter zuhilfenahme der anderen Klassen aus und stellt Output Daten an den OutputPorts bereit
	@Override
	public void doWork() throws OperatorException {
		//Daten des InputPorts erhalten und an den OutputPort für das ExampleSet übergeben
		ExampleSet inputData = exampleSetInputPort.getData(ExampleSet.class);
		exampleSetOutputPort.deliver(inputData);
		ExampleSet trainingSet = (ExampleSet) inputData.clone();
		
		//RandomGenerator wird für die Generierung der "Zufallszahlen" verwendet.
		//Es werden für jede Ausführung die gleichen Zufallszahlen in Abhängigkeit vom RandomSeed erzeugt.
		RandomGenerator randomGenerator = RandomGenerator.getRandomGenerator(this);
		
		//Selektierte Operatorparameter erhalten
		ppc = getParameterAsInt(PARAMETER_PROTOTYPES_PER_CLASS);
		iterations = getParameterAsInt(PARAMETER_ITERATIONS);
		updateRate = getParameterAsDouble(PARAMETER_UPDATE_RATE);
		decayFactor = getParameterAsDouble(PARAMETER_DECAY_FACTOR);
		sigma = getParameterAsDouble(PARAMETER_SIGMA);
		costLog = getParameterAsBoolean(PARAMETER_COST_LOG);
		
		//Attribute von InputData erhalten und zu den DebugAttributen hinzufügen
		Iterator<Attribute> attributes = inputData.getAttributes().allAttributes();
		debugAtt = inputData.getAttributes();
		
		//Attribute zwischenspeichern
		ArrayList<Attribute> attributesList = new ArrayList<>();
		for ( Iterator<Attribute> i = attributes; attributes.hasNext(); ){
			attributesList.add((Attribute) i.next().clone());
		}
		Attribute[] attributesArr = attributesList.toArray(new Attribute[attributesList.size()]);
		String[] attributeStr = new String[attributesList.size()];
		int attrI = 0;
		for(Attribute attr : attributesArr) {
			attributeStr[attrI] = attr.getName();
			LogService.getRoot().log(Level.INFO, "DEBUG:            attributes from attrStr :  " + attr.getName());
			attrI++;
		}
		attributes = inputData.getAttributes().allAttributes();
		Attribute label = inputData.getAttributes().getLabel();
		
		//Prüfen ob ein Label spezifiziert wurde
		if(label != null) {
			if(debugMode) LogService.getRoot().log(Level.INFO, "DEBUG:            Label =  " + label.getName());
		}
		else {
			throw new OperatorException("Could not identify the label inside the input data. Assign the role for your classification attribute.");
		}
		
		//Alle unterschiedlichen Labels speichern
		HashSet<String> classes = new HashSet<>();
		Iterator<Example> trainingSetIter = trainingSet.iterator();
		for(Iterator<Example> i = trainingSetIter; i.hasNext();) {
			classes.add((String)i.next().getValueAsString(label));
		}	
		
		//Prototypen initialisieren
		ExampleSet initialPrototypes = null;
		List<Attribute> listOfAtts = new LinkedList<>();
		boolean lbl = false;
		Attribute cbLbl = null;
		
		for(Attribute att : attributesList) {
			Attribute newAtt = AttributeFactory.createAttribute(att);
			if(att.getName() == inputData.getAttributes().getLabel().getName()) lbl = true;
			LogService.getRoot().log(Level.INFO, "Attribute Type " + att.getValueType());
			if(lbl) cbLbl = newAtt;
			lbl = false;
			listOfAtts.add(newAtt);
		}
		// ExampleSetBuilder wird für die Erzeugung neuer ExampleSets benötigt
		ExampleSetBuilder builder = ExampleSets.from(listOfAtts);
		builder.withRole(cbLbl,  "label");
		
		// Examples werden mit double Arrays erzeugt
		double[] doubleArray = new double[attributesList.size()];
		// Für jedes Label wird ein zufälliges Example aus InputData gezogen (mit entsprechendem Label)
		for(String cls : classes) {
			if(debugMode) LogService.getRoot().log(Level.INFO, "Alle Meine Klassen: " + cls);
			int i = 1;
			for(Example randomExample = trainingSet.getExample(randomGenerator.nextInt(trainingSet.size()-1));i <= ppc;randomExample = trainingSet.getExample(randomGenerator.nextInt(trainingSet.size()-1)))
			{
				if(randomExample.getValueAsString(label) == cls) {
					int arrIndex = 0;
					
					for(String att : attributeStr) {
						doubleArray[arrIndex] = randomExample.getValue(randomExample.getAttributes().get(att));
						arrIndex++;
					}
					if(debugMode) LogService.getRoot().log(Level.INFO, " arrayPrint" + Arrays.toString(doubleArray));
					builder.addRow(doubleArray);
					i++;
				}
			}
		}
					
		// Prototypen erzeugen
		initialPrototypes = builder.build();
		
		// Special Attributes müssen manuell übertragen werden
		for(Iterator<Attribute> i = inputData.getAttributes().allAttributes(); i.hasNext(); )
		{
		  Attribute att = i.next();
		  for(Example example : initialPrototypes) {
			  example.getAttributes().setSpecialAttribute(example.getAttributes().get(att.getName()), inputData.getAttributes().getRole(att).getSpecialName());
		  }
		}

		LogService.getRoot().log(Level.INFO, "INITIAL PROTOTYPES");
		for(Example example : initialPrototypes) {
			printExample(example);
		}
		
		// Erzeuge RSLVQModel für das Training
		RSLVQModel model = new RSLVQModel(initialPrototypes, sigma, ppc, updateRate, decayFactor, costLog);
		
		// Trainingsphase durchführen (Prototypen werden in RSLVQModel gespeichert)
		model.fit(trainingSet,iterations, randomGenerator);
		
		// Erhalte die trainierten Prototypen
		ExampleSet prototypes = model.getPrototypes();
		
		// Erhalte das trainierte Model
		PredictionModel rapidminerModel = model.getPredictionModel();

		// Output an den Ports zur Verfügung stellen
		if(costLog) costProgressOutputPort.deliver(model.getCostProgress());
		modelOutputPort.deliver(rapidminerModel);
		prototypesOutputPort.deliver(prototypes);
	}

	protected MetaData modifyPrototypeOutputMetaData(ExampleSetMetaData metaData)
			throws UndefinedParameterError {
		try { 
			metaData.setNumberOfExamples(getNumberOfPrototypesMetaData());
		} catch (UndefinedParameterError e){
			metaData.setNumberOfExamples(new MDInteger());
		}
		return metaData;
	}

	protected MDInteger getNumberOfPrototypesMetaData() throws UndefinedParameterError {
		int num = getParameterAsInt(PARAMETER_PROTOTYPES_PER_CLASS);
		return new MDInteger(num);                        
	}
	
	// Gibt an welche Datentypen vom Operator unterstützt werden
	@Override
	public boolean supportsCapability(OperatorCapability capability) {
		switch (capability) {
		case POLYNOMINAL_ATTRIBUTES:
		case NUMERICAL_ATTRIBUTES:
		case POLYNOMINAL_LABEL:
		case BINOMINAL_LABEL:
			return true;
		default:
			return false;
		}
	}
	
	// Stellt die Operatorenparameter zur Verfügung
	public List<ParameterType> getParameterTypes(){
		List<ParameterType> types = super.getParameterTypes();
		
		ParameterType type; 

		type = new ParameterTypeInt(PARAMETER_PROTOTYPES_PER_CLASS, "Number of Prototypes per Class", 1, Integer.MAX_VALUE , this.ppc);
        type.setExpert(false);
        types.add(type);

        type = new ParameterTypeInt(PARAMETER_ITERATIONS, "Number of iterations", 0, Integer.MAX_VALUE, this.iterations);
        type.setExpert(false);
        types.add(type);

        type = new ParameterTypeDouble(PARAMETER_UPDATE_RATE, "Update rate", 0.0000000001, 0.9999999999, this.updateRate);
        type.setExpert(false);
        types.add(type);
        
        type = new ParameterTypeDouble(PARAMETER_DECAY_FACTOR, "Value of update rate", 0.0000000001, 0.9999999999, this.decayFactor);
        type.setExpert(false);
        types.add(type);
        
        type = new ParameterTypeDouble(PARAMETER_SIGMA, "Defines the Sigma", 0.0000000001, 0.9999999999, this.sigma);
        type.setExpert(false);
        types.add(type);
        
        type = new ParameterTypeBoolean(PARAMETER_COST_LOG, "Show Error for each epoch on Log-Screen", false);
        type.setExpert(false);
        types.add(type);
        
        Collection<ParameterType> parameters = RandomGenerator.getRandomGeneratorParameters(this);
        
        types.addAll(parameters);
		return types;
	}
	
	//Hilfsmethode für das Debugging. Gibt alle Attribute des übergebenen Examples auf der Log Console in RapidMiner aus.
	public static void printExample(Example example) {
		String out = new String();
		ArrayList<String> attNames = new ArrayList<>();
		for(Iterator<Attribute> debugAtt = example.getAttributes().allAttributes() ; debugAtt.hasNext();) {
			attNames.add(debugAtt.next().getName());
		}
		for(String name : attNames) {
			out += name + ": " + example.getValue(example.getAttributes().get(name)) + "  -  ";
		}
		LogService.getRoot().log(Level.INFO, "example: " + out + "typeOfDataRow" + example.getDataRow().getType());
	}
	
    public InputPort getExampleSetInputPort() {
        return exampleSetInputPort;
    }

    public OutputPort getExampleSetOutputPort() {
        return exampleSetOutputPort;
    }

    public OutputPort getProtoOutputPort(){
        return prototypesOutputPort;
    }
    
    public OutputPort getModelOutputPort(){
        return modelOutputPort;
    }
    
    public OutputPort getProgressOutputPort(){
        return costProgressOutputPort;
    }
	
    // Erstellt Transformationsregeln zwischen Input und Output Daten
	protected void addPrototypeTransformationRule(){
		getTransformer().addRule(new PassThroughRule(exampleSetInputPort, prototypesOutputPort, true) {
			@Override
			public MetaData modifyMetaData(MetaData metaData) {
				if (metaData instanceof ExampleSetMetaData) {
					try {
						return RSLVQ_Operator.this.modifyPrototypeOutputMetaData((ExampleSetMetaData) metaData);
					} catch (UndefinedParameterError ex) {
						return metaData;
					}
				} else {
					return metaData;
				}
			}
		});
	}      
}
