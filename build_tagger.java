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
import java.util.ArrayList;
import java.util.List;

// <Enumeration Description Here>
enum SmoothingTechnique {LAPLACE, WITTENBELL, KNESERNEY}

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
        List<String[]> trainCorpus = null;
        List<String[]> devCorpus = null;

        // Read sentences from files
        if (args.length >= 3) {
            FileHandler trainFile = new FileHandler(args[0]);
            trainFile.readFile();
            trainCorpus = trainFile.getFileAsCorpus();

            FileHandler devFile = new FileHandler((args[1]));
            devFile.readFile();
            devCorpus = devFile.getFileAsCorpus();

            FileHandler modelFile = new FileHandler(args[2]);
        } else {
            System.err.println("Incorrect number of parameters.");
            System.exit(-1);
        }

        /* TODO: Create list of conditions (smoothing technique, interpolation) */

        /* TODO: For every condition, create a model, train and test to get best condition set */

        /* TODO: Perform cross-validation for best model to obtain accuracy */
    }
}

// <Class Description Here>
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

    public String getFileAsString() {
        if (existingFileData != null) {
            return new String(existingFileData);
        } else {
            return null;
        }
    }

    public String[] getFileAsSentences() {
        if (existingFileData != null) {
            String linesString = new String(existingFileData);
            return linesString.split("[\\r?\\n]+");
        } else {
            return null;
        }
    }

    public List<String[]> getFileAsCorpus() {
        if (existingFileData != null) {
            String linesString = new String(existingFileData);
            String[] sentences = linesString.split("[\\r?\\n]+");
            List<String[]> corpus = new ArrayList<String[]>();
            for (String sentence : sentences) {
                corpus.add(sentence.split(" "));
            }
            return corpus;
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

// <Class Description Here>
class Model {
    // Smoothing technique to be applied on P(tag|prev-tag)
    SmoothingTechnique smoothingMode = SmoothingTechnique.LAPLACE;
    // Interpolation to be applied on P(tag|prev-tag)
    double interpolationRatio = 1;

    // Track emission: P(word|tag)
    private List<Double[]> emissionProbMatrix = null;
    // Track path transition: max[(t-1-viterbis) * P(tag|prev-tag)] * P(word|tag)
    private List<Double[]> pathProbMatrix = null;
    // Track states (tags): previous tag -> argmax[(t-1-viterbis) * P(tag|prev-tag)]
    private List<Double[]> backpointerMatrix = null;

    // <Abstract Interface Description Here>
    private interface Smoothing {
        public double bigramApprox();
    }

    // <Abstract Class Description Here>
    private abstract class SmoothScheme implements Smoothing {
        List<String[]> corpus = null;
        String wordTag = null;
        String prevWordTag = null;

        public SmoothScheme(List<String[]> corpus, String wordTag, String prevWordTag) {
            super();
            this.corpus = corpus;
            this.wordTag = wordTag;
            this.prevWordTag = prevWordTag;
        }
    }

    private class Laplace extends SmoothScheme {
        int laplaceFactor = 1;

        public Laplace(List<String[]> corpus, String wordTag, String prevWordTag) {
            super(corpus, wordTag, prevWordTag);
        }

        public Laplace(List<String[]> corpus, String wordTag, String prevWordTag, int laplaceFactor) {
            super(corpus, wordTag, prevWordTag);
            this.laplaceFactor = laplaceFactor;
        }

        public double bigramApprox() {
            /* TODO: Laplace smoothing logic */
            return 0;
        }
    }

    private class WittenBell extends SmoothScheme {
        int seenTypes;
        int unseenTypes;

        public WittenBell(List<String[]> corpus, String wordTag, String prevWordTag) {
            super(corpus, wordTag, prevWordTag);
            /* TODO: Calculate and initiate seen and unseen types from corpus */
        }

        public double bigramApprox() {
            /* TODO: Witten Bell smoothing logic */
            return 0;
        }
    }

    private class KneserNey extends SmoothScheme {
        double absoluteDiscount;
        double backoffFactor;

        public KneserNey(List<String[]> corpus, String wordTag, String prevWordTag) {
            super(corpus, wordTag, prevWordTag);
            /* TODO: Calculate and initiate absoluteDiscount and backoffFactor */
        }

        public double bigramApprox() {
            /* TODO: Kneser-Ney smoothing logic */
            return 0;
        }
    }

    public Model() {
        /* TODO: Use default smoothing scheme and interpolation */
    }

    public Model(SmoothingTechnique smoothingScheme) {
        /* TODO: Assign smoothing scheme and use default interpolation */
    }

    public Model(SmoothingTechnique smoothingScheme, double interpolationRatio) {
        /* TODO: Assign smoothing scheme and interpolation */
    }

    public void train(List<String[]> trainingCorpus) {
        /* TODO: Call init(s) then apply viterbi algorithm based on conditions */
    }

    public void test(List<String[]> testCorpus) {
        /* TODO: Strip test corpus for blind testing */
        /* TODO: Apply backpointer to test for tagging */
        /* TODO: Compare and print tagged line and accuracy */
    }

    public double getAccuracy(int n) {
        /* TODO: Perform n-fold cross validation on training corpus */
        return 0;
    }

    private void initEmissionProbMatrix() {
        /* TODO: Fill with calculation from corpus */
    }

    private void initPathProbMatrix() {
        /* TODO: Create sized blank matrix */
    }

    private void initBackpointerMatrix() {
        /* TODO: Create sized blank matrix */
    }

    private List<String[]> getStrippedCorpus(List<String[]> taggedCorpus) {
        /* TODO: Strip corpus of all tags */
        return null;
    }
}