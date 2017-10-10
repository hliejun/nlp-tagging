/**
 * build_tagger trains the POS tagger with sents.train
 * and tunes the POS tagger parameters with sents.devt,
 * then writes the output statistics to model_file
 *
 * @author Huang Lie Jun (A0123994W)
 * @version 1.0
 * @since 2017-10-08
 */
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

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
        String[] trainSentences = null;
        String[] devSentences = null;

        // Read sentences from files
        if (args.length >= 3) {
            FileHandler trainFile = new FileHandler(args[0]);
            trainFile.readFile();
            trainSentences = trainFile.getFileLines();
            System.out.println(trainSentences.length);

            FileHandler devFile = new FileHandler((args[1]));
            devFile.readFile();
            devSentences = devFile.getFileLines();
            System.out.println(devSentences.length);

            FileHandler modelFile = new FileHandler(args[2]);
        } else {
            System.err.println("Incorrect number of parameters.");
            System.exit(-1);
        }

        // For every smoothing technique,
        //     1. train (build model, on train sentences)
        //     2. tune (test, on development sentences)
        // to identify best smoothing scheme/model

        // Perform cross-validation on training sentences to obtain accuracy
    }
}

// <Function Description Here>
class FileHandler {
    Path filePath;
    byte[] existingFileData;

    public FileHandler(String pathString) {
        filePath = Paths.get(pathString);
    }

    public void readFile() {
        if (Files.exists(filePath)) {
            try {
                existingFileData = Files.readAllBytes(filePath);
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            System.err.println("File to be read does not exist.");
        }
    }

    public String getFileString() {
        if (existingFileData != null) {
            return new String(existingFileData);
        } else {
            return null;
        }
    }

    public String[] getFileLines() {
        if (existingFileData != null) {
            String linesString = new String(existingFileData);
            return linesString.split("[\\r?\\n]+");
        } else {
            return null;
        }
    }

    public void writeFile(String writeData) {
        try {
            byte[] newFileData = writeData.getBytes("utf-8");
            Files.write(filePath, newFileData, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

/* TODO: Run Viterbi on training sentences to obtain model tables */
/* TODO: Interpolation */
class Model {

}

/* TODO: Extend to different smoothing techniques (Laplace, Witten Bell, Kneser-Ney) */
class SmoothScheme {
    // Variables:
    //      (String) current
    //      (String) prev
    //      (String[]) corpus
    //      (String[]) currentSentence
    //      [Laplace] smoothingFactor
    //      [Witten Bell] seenTypes
    //      [Witten Bell] unseenTypes
    //      [Kneser-Ney] absoluteDiscount
    //      [Kneser-Ney] backoffFactor

    // Methods:
    //      (interface) getBigramPrediction (scheme formula)
    //      [Witten Bell] initTypeCount
    //      [Kneser-Ney] initDiscount
    //      [Kneser-Ney] initBackoff
}

/* TODO: Cross validate using optimal model and smoother on training data */
class CrossValidator {

}