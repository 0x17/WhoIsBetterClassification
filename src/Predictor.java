import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeSelection;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class Predictor {

    static class InstancesPair {
        Instances train, validation;
    }

    private static InstancesPair split(Instances instances, float train_split) {
        InstancesPair pair = new InstancesPair();
        int split_ix = Math.round(instances.size() * train_split);
        pair.train = new Instances(instances, 0, split_ix);
        pair.validation = new Instances(instances, split_ix, instances.size()-split_ix);
        return pair;
    }

    private static Instances attributeSelectionMechanism(Instances instances, boolean bypass) throws Exception {
        if(bypass) return instances;
        AttributeSelection as = new AttributeSelection();
        ASSearch asSearch = ASSearch.forName("weka.attributeSelection.BestFirst", new String[]{"-D", "1", "-N", "4"});
        as.setSearch(asSearch);
        ASEvaluation asEval = ASEvaluation.forName("weka.attributeSelection.CfsSubsetEval", new String[]{"-L"});
        as.setEvaluator(asEval);
        as.SelectAttributes(instances);
        return as.reduceDimensionality(instances);
    }

    public static void main(String[] args) throws Exception {
        final float train_split = 0.95f;
        Instances namedInstances = ConverterUtils.DataSource.read("whoisbetter_with_names.arff");
        namedInstances.randomize(new Random(1));

        Instances instances = new Instances(namedInstances);
        instances.deleteStringAttributes();
        instances.setClassIndex(instances.numAttributes()-1);
        instances = attributeSelectionMechanism(instances, true);

        InstancesPair namedPair = split(namedInstances, train_split);
        InstancesPair pair = split(instances, train_split);

        //Classifier clf = AbstractClassifier.forName("weka.classifiers.functions.Logistic", new String[]{"-R", "0.057140274761388915"});
        Classifier clf = AbstractClassifier.forName("weka.classifiers.trees.RandomForest", Utils.splitOptions("-P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1"));

        clf.buildClassifier(pair.train);
        System.out.println(clf);

        Evaluation eval = new Evaluation(pair.train);

        final boolean predictionsOnValidationData = true;

        Instances predictionSet, namedPredictionSet;

        if(predictionsOnValidationData) {
            predictionSet = pair.validation;
            namedPredictionSet = namedPair.validation;
        } else {
            predictionSet = instances;
            namedPredictionSet = namedInstances;
        }

        eval.evaluateModel(clf, predictionSet);
        System.out.println(eval.toSummaryString("Results:\n", false));

        List<String> instanceNames = namedPredictionSet.stream().map(inst -> inst.stringValue(0)).collect(Collectors.toList());
        ArrayList<Prediction> preds = eval.predictions();
        StringBuilder sb = new StringBuilder();
        for(int i=0; i<preds.size(); i++) {
            Prediction pred = preds.get(i);
            sb.append(instanceNames.get(i));
            sb.append(";");
            sb.append(pred.predicted());
            sb.append(";");
            sb.append(pred.actual());
            sb.append('\n');
        }

        Files.write(Paths.get("predictions_models_onlyvalidation.csv"), sb.toString().getBytes());
    }

}
