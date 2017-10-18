/**
 * build_tagger trains the POS tagger with sents.train
 * and tunes the POS tagger parameters with sents.devt,
 * then writes the output statistics to model_file
 *
 * @author Huang Lie Jun (A0123994W)
 * @version 1.0
 * @since 2017-10-08
 */
import java.util.*;

public class build_tagger {
    /**
     * This function will train a HMM model using training sentences,
     * select the optimal smoothing scheme for the trained model using
     * development sentences, perform a 10-fold validation on training
     * sentences to acquire averaged accuracy and write model results
     * to file for testing and actual tagging.
     *
     * @param trainFile File path to training sentences
     * @param devFile File path to development sentences
     * @param modelFile File path to write model data (params. and prob.)
     */
    public static void main(String[] args) {
        FileHandler trainFile, devFile, modelFile = null;
        List<String[]> trainCorpus = null;
        List<String[]> devCorpus = null;
        if (args.length >= 3) {
            trainFile = new FileHandler(args[0]);
            trainFile.readFile();
            trainCorpus = trainFile.getFileAsCorpus();
            devFile = new FileHandler((args[1]));
            devFile.readFile();
            devCorpus = devFile.getFileAsCorpus();
            modelFile = new FileHandler(args[2]);
        } else {
            System.err.println("Incorrect number of parameters.");
            System.exit(-1);
        }
        Model posModel = new Model();
        posModel.train(trainCorpus);
        posModel.tune(devCorpus);
        float validatedAccuracy = posModel.crossValidate(trainCorpus, 10);
        System.out.println("Cross-validation accuracy of trained model: " + (validatedAccuracy * 100) + "%");
        if (modelFile != null) {
            modelFile.writeFile(posModel);
        }
    }
}