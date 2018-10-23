import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;

public class Main {

    private static Instances loadData(String filename, boolean removeNames) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setFieldSeparator(";");
        loader.setStringAttributes("first");
        loader.setNominalAttributes("last");
        loader.setSource(new File(filename));
        Instances data = loader.getDataSet();
        if(removeNames)
            data.deleteStringAttributes();

        data.setClassIndex(data.numAttributes() - 1);

        /*AddID filter = new AddID();
        filter.setInputFormat(data);
        return Filter.useFilter(data, filter);*/
        return data;
    }

    private static void saveAsArff(Instances instances, String filename) throws Exception {
        File f = new File(filename);
        ArffSaver saver = new ArffSaver();
        saver.setInstances(instances);
        saver.setDestination(f);
        saver.writeBatch();
    }

    private static void saveAsArffAlt(Instances instances, String filename) throws Exception {
        ConverterUtils.DataSink.write(filename, instances);
    }

    public static void main(String... args) throws Exception {
        String folder = "combined_j30_upto_120_rgen";

        Instances data = loadData(folder+"/whoisbetter.csv", true);
        saveAsArffAlt(data, "whoisbetter.arff");
        Files.copy(Paths.get("whoisbetter.arff"), Paths.get("/Users/andreschnabel/whoisbetter.arff"), StandardCopyOption.REPLACE_EXISTING);

        Instances dataWithNames = loadData(folder+"/whoisbetter.csv", false);
        saveAsArffAlt(dataWithNames, "whoisbetter_with_names.arff");
        Files.copy(Paths.get("whoisbetter_with_names.arff"), Paths.get("/Users/andreschnabel/whoisbetter_with_names.arff"), StandardCopyOption.REPLACE_EXISTING);
    }

}
