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
import java.util.*;

enum Technique {LAPLACE, WITTENBELL, KNESERNEY}
enum Type {WORD, TAG, BOTH}

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
        Model posModel = new Model();
        posModel.train(trainCorpus);
        posModel.tune(devCorpus);
        double validatedAccuracy = posModel.crossValidate(trainCorpus, 10);
        System.out.println("Cross-validation accuracy of trained model: " + (validatedAccuracy * 100) + "%");
        Technique bestTechnique = posModel.getBestTechnique();
        HashMap<String, Double> modelTransitionMatrix = posModel.getTransitionProbMatrix();
        HashMap<String, Double> modelEmissionMatrix = posModel.getEmissionProbMatrix();
        HashMap<String, Integer> wordFreqMatrix = posModel.getWordFreq();
        HashMap<String, Integer> tagFreqMatrix = posModel.getTagFreq();
        HashMap<String, Integer> wordTagFreqMatrix = posModel.getWordTagFreq();
        HashMap<String, Integer> prevCurrTagFreqMatrix = posModel.getPrevCurrTagFreq();

        // TODO: Write model attributes to modelFile
    }
}

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

class Model {
    private Technique smoothingMode = Technique.LAPLACE;
    private HashMap<String, Integer> wordFreq, tagFreq, wordTagFreq, prevCurrTagFreq;
    private HashMap<String, Double> emissionProbMatrix, transitionProbMatrix;
    private List<String> uniqueWords, uniqueTags;

    public Model() {
        super();
    }

    public void train(List<String[]> trainingCorpus) {
        tagFreq = new HashMap<String, Integer>();
        wordFreq = new HashMap<String, Integer>();
        wordTagFreq = new HashMap<String, Integer>();
        prevCurrTagFreq = new HashMap<String, Integer>();
        String prev = "";
        String curr = "";
        String[] currWordTag;
        String prevWord, prevTag, currWord, currTag, prevCurrSymbol, wordTagSymbol;
        for (String[] sentence : trainingCorpus) {
            for (int index = 0; index < sentence.length; index++) {
                curr = sentence[index];
                currWordTag = splitElement(curr);
                currWord = currWordTag[0];
                if (wordFreq.get(currWord) == null) {
                    wordFreq.put(currWord, 1);
                } else {
                    wordFreq.put(currWord, wordFreq.get(currWord) + 1);
                }
                if (wordTagFreq.get(curr) == null) {
                    wordTagFreq.put(curr, 1);
                } else {
                    wordTagFreq.put(curr, wordTagFreq.get(curr) + 1);
                }
                currTag = currWordTag[1];
                if (index == 0 && tagFreq.get("<s>") == null) {
                    tagFreq.put("<s>", 1);
                } else if (index == 0) {
                    tagFreq.put("<s>", tagFreq.get("<s>") + 1);
                }
                if (tagFreq.get(currTag) == null) {
                    tagFreq.put(currTag, 1);
                } else {
                    tagFreq.put(currTag, tagFreq.get(currTag) + 1);
                }
                if (index == 0) {
                    prevTag = "<s>";
                } else {
                    prev = sentence[index - 1];
                    prevTag = splitElement(prev)[1];
                }
                prevCurrSymbol = prevTag + "/" + currTag;
                if (prevCurrTagFreq.get(prevCurrSymbol) == null) {
                    prevCurrTagFreq.put(prevCurrSymbol, 1);
                } else {
                    prevCurrTagFreq.put(prevCurrSymbol, prevCurrTagFreq.get(prevCurrSymbol) + 1);
                }
            }
        }
        uniqueWords = new ArrayList<String>(wordFreq.keySet());
        uniqueTags = new ArrayList<String>(tagFreq.keySet());
        Collections.sort(uniqueWords);
        Collections.sort(uniqueTags);
        transitionProbMatrix = new HashMap<String, Double>();
        emissionProbMatrix = new HashMap<String, Double>();
        for (int row = 0; row < uniqueTags.size(); row++) {
            currTag = uniqueTags.get(row);
            for (int col = 0; col < uniqueTags.size(); col++) {
                prevTag = uniqueTags.get(col);
                prevCurrSymbol = prevTag + "/" + currTag;
                Integer prevCurrCount = prevCurrTagFreq.get(prevCurrSymbol);
                Integer prevTagCount = tagFreq.get(prevTag);
                prevCurrCount = (prevCurrCount != null) ? prevCurrCount : 0;
                prevTagCount = (prevTagCount != null) ? prevTagCount : 0;
                transitionProbMatrix.put(prevCurrSymbol, (double)prevCurrCount / (double)prevTagCount);
            }
        }
        for (int row = 0; row < uniqueWords.size(); row++) {
            currWord = uniqueWords.get(row);
            for (int col = 0; col < uniqueTags.size(); col++) {
                currTag = uniqueTags.get(col);
                wordTagSymbol = currWord + "/" + currTag;
                Integer wordTagCount = wordTagFreq.get(wordTagSymbol);
                Integer tagCount = tagFreq.get(currTag);
                wordTagCount = (wordTagCount != null) ? wordTagCount : 0;
                tagCount = (tagCount != null) ? tagCount : 0;
                emissionProbMatrix.put(wordTagSymbol, (double)wordTagCount / (double)tagCount);
            }
        }
    }

    public double test(List<String[]> testCorpus, Technique smoothingScheme) {
        int correct = 0;
        int total = 0;
        SmoothScheme smoother = null;
        List<String[]> untaggedTestCorpus = getStrippedCorpus(testCorpus);
        HashMap<String, Integer> uniqueWordsTested = new HashMap<String, Integer>();
        for (String[] sentence : untaggedTestCorpus) {
            for (String word : sentence) {
                if (uniqueWordsTested.get(word) == null) {
                    uniqueWordsTested.put(word, 1);
                } else {
                    uniqueWordsTested.put(word, uniqueWordsTested.get(word) + 1);
                }
            }
        }
        Set<String> uniqueTestWords = uniqueWordsTested.keySet();
        Set<String> uniqueSeenWords = wordFreq.keySet();
        Set<String> uniqueUnseenWords = new HashSet<String>(uniqueTestWords);
        uniqueUnseenWords.removeAll(uniqueSeenWords);
        switch (smoothingScheme) {
            case LAPLACE:
                smoother = new Laplace(wordFreq, tagFreq, wordTagFreq, prevCurrTagFreq, 1);
                break;
            case WITTENBELL:
                int seen = uniqueSeenWords.size();
                int unseen = uniqueUnseenWords.size();
                smoother = new WittenBell(wordFreq, tagFreq, wordTagFreq, prevCurrTagFreq, seen, unseen);
                break;
            default:
                smoother = new Laplace(wordFreq, tagFreq, wordTagFreq, prevCurrTagFreq, 1);
                break;
        }
        for (int sentenceIndex = 0; sentenceIndex < untaggedTestCorpus.size(); sentenceIndex++) {
            String[] sentence = untaggedTestCorpus.get(sentenceIndex);
            String[] taggedSentence = testCorpus.get(sentenceIndex);
            double[][] pathProbMatrix = new double[uniqueTags.size() + 1][sentence.length];
            int[][] backpointerMatrix = new int[uniqueTags.size() + 1][sentence.length];
            double maxPathValue = -1;
            int bestPrevTagIndex = -1;
            for (int wordIndex = 0; wordIndex < sentence.length; wordIndex++) {
                String currentWord = sentence[wordIndex];
                for (int tagIndex = 0; tagIndex < uniqueTags.size(); tagIndex++) {
                    Double alpha, beta;
                    String currentTag = uniqueTags.get(tagIndex);
                    if (currentTag == "<s>") {
                        continue;
                    } else if (wordIndex == 0) {
                        alpha = transitionProbMatrix.get("<s>/" + currentTag);
                        alpha = (alpha != null) ? alpha : smoother.getBigramTransition("<s>/" + currentTag);
                        beta = emissionProbMatrix.get(currentWord + "/" + currentTag);
                        beta = (beta != null) ? beta : smoother.getBigramEmission(currentWord + "/" + currentTag);
                        pathProbMatrix[tagIndex][wordIndex] = alpha * beta;
                        backpointerMatrix[tagIndex][wordIndex] = -1;
                    } else {
                        bestPrevTagIndex = 0;
                        maxPathValue = 0.0d;
                        for (int prevTagIndex = 0; prevTagIndex < uniqueTags.size(); prevTagIndex++) {
                            String prevTag = uniqueTags.get(prevTagIndex);
                            alpha = transitionProbMatrix.get(prevTag + "/" + currentTag);
                            alpha = (alpha != null) ? alpha : smoother.getBigramTransition(prevTag + "/" + currentTag);
                            double value = pathProbMatrix[prevTagIndex][wordIndex - 1] * alpha;
                            if (value >= maxPathValue) {
                                maxPathValue = value;
                                bestPrevTagIndex = prevTagIndex;
                            }
                        }
                        beta = emissionProbMatrix.get(currentWord + "/" + currentTag);
                        beta = (beta != null) ? beta : smoother.getBigramEmission(currentWord + "/" + currentTag);
                        pathProbMatrix[tagIndex][wordIndex] = maxPathValue * beta;
                        backpointerMatrix[tagIndex][wordIndex] = bestPrevTagIndex;
                    }
                }
            }
            int bestTagIndex = 0;
            maxPathValue = 0.0d;
            for (int tagIndex = 0; tagIndex < uniqueTags.size(); tagIndex++) {
                String tag = uniqueTags.get(tagIndex);
                if (tag == "<s>") {
                    continue;
                }
                double pathValue = pathProbMatrix[tagIndex][sentence.length - 1];
                if (pathValue >= maxPathValue) {
                    maxPathValue = pathValue;
                    bestTagIndex = tagIndex;
                }
            }
            pathProbMatrix[uniqueTags.size()][sentence.length - 1] = maxPathValue;
            backpointerMatrix[uniqueTags.size()][sentence.length - 1] = bestTagIndex;
            int prevStateIndex = bestTagIndex;
            int prevSequenceIndex = sentence.length - 1;
            List<String> prediction = new ArrayList<String>();
            while (prevStateIndex != -1 && prevSequenceIndex >= 0) {
                String tag = uniqueTags.get(prevStateIndex);
                String word = sentence[prevSequenceIndex];
                prediction.add(0, word + "/" + tag);
                prevStateIndex = backpointerMatrix[prevStateIndex][prevSequenceIndex];
                prevSequenceIndex -= 1;
            }
            for (int predictionIndex = 0; predictionIndex < prediction.size(); predictionIndex++) {
                if (prediction.get(predictionIndex).equals(taggedSentence[predictionIndex])) {
                    correct += 1;
                }
                total += 1;
            }
        }
        return (double)correct / (double)total;
    }

    public void tune(List<String[]> testCorpus) {
        Technique[] techniques = new Technique[1];
        techniques[0] = Technique.LAPLACE;
        double bestAccuracy = 0;
        double currentAccuracy = 0;
        for (Technique technique : techniques) {
            currentAccuracy = this.test(testCorpus, technique);
            if (currentAccuracy >= bestAccuracy) {
                bestAccuracy = currentAccuracy;
                this.smoothingMode = technique;
            }
        }
        System.out.println("Best Technique: " + this.smoothingMode);
        System.out.println("Best Accuracy: " + bestAccuracy * 100 + "%");
    }

    public double crossValidate(List<String[]> trainingCorpus, int n) {

        // TODO: Perform n-fold cross validation on training corpus

        return 0;
    }

    public Technique getBestTechnique() {
        return this.smoothingMode;
    }

    public HashMap<String, Double> getTransitionProbMatrix() {
        return this.transitionProbMatrix;
    }

    public HashMap<String, Double> getEmissionProbMatrix() {
        return this.emissionProbMatrix;
    }

    public HashMap<String, Integer> getWordFreq() {
        return this.wordFreq;
    }

    public HashMap<String, Integer> getTagFreq() {
        return this.tagFreq;
    }

    public HashMap<String, Integer> getWordTagFreq() {
        return this.wordTagFreq;
    }

    public HashMap<String, Integer> getPrevCurrTagFreq() {
        return this.prevCurrTagFreq;
    }

    private interface Smoothing {
        public double getBigramTransition(String prevCurrTag);
        public double getBigramEmission(String wordTag);
    }

    private abstract class SmoothScheme implements Smoothing {
        private HashMap<String, Integer> wordFreq, tagFreq, wordTagFreq, prevCurrTagFreq;

        public SmoothScheme(HashMap<String, Integer> wordFreq, HashMap<String, Integer> tagFreq, HashMap<String, Integer> wordTagFreq, HashMap<String, Integer> prevCurrTagFreq) {
            super();
            this.wordFreq = wordFreq;
            this.tagFreq = tagFreq;
            this.wordTagFreq = wordTagFreq;
            this.prevCurrTagFreq = prevCurrTagFreq;
        }
    }

    private class Laplace extends SmoothScheme {
        int laplaceFactor;

        public Laplace(HashMap<String, Integer> wordFreq, HashMap<String, Integer> tagFreq, HashMap<String, Integer> wordTagFreq, HashMap<String, Integer> prevCurrTagFreq) {
            super(wordFreq, tagFreq, wordTagFreq, prevCurrTagFreq);
            this.laplaceFactor = 1;
        }

        public Laplace(HashMap<String, Integer> wordFreq, HashMap<String, Integer> tagFreq, HashMap<String, Integer> wordTagFreq, HashMap<String, Integer> prevCurrTagFreq, int laplaceFactor) {
            super(wordFreq, tagFreq, wordTagFreq, prevCurrTagFreq);
            this.laplaceFactor = laplaceFactor;
        }

        public double getBigramTransition(String prevCurrTag) {
            String prevTag = splitElement(prevCurrTag)[0];
            Integer prevCurrTagCount = prevCurrTagFreq.get(prevCurrTag);
            prevCurrTagCount = (prevCurrTagCount != null) ? prevCurrTagCount : 0;
            Integer prevTagCount = tagFreq.get(prevTag);
            prevTagCount = (prevTagCount != null) ? prevTagCount : 0;
            Integer uniqueTagCount = tagFreq.size();
            uniqueTagCount = (uniqueTagCount != null) ? uniqueTagCount : 0;
            return ((double)prevCurrTagCount + 1) / ((double)prevTagCount + (double)uniqueTagCount);
        }

        public double getBigramEmission(String wordTag) {
            Integer wordTagCount = wordTagFreq.get(wordTag);
            wordTagCount = (wordTagCount != null) ? wordTagCount : 0;
            Integer tagCount = tagFreq.get(splitElement(wordTag)[1]);
            tagCount = (tagCount != null) ? tagCount : 0;
            Integer uniqueTagCount = tagFreq.size();
            uniqueTagCount = (uniqueTagCount != null) ? uniqueTagCount : 0;
            return ((double)wordTagCount + 1) / ((double)tagCount + (double)uniqueTagCount);
        }

        // KIV
        public double getUnknownWordBigramEmission(String wordTag, String prevWordTag) {
            // P(word|tag) = P(word|tag) * P(capital-word|tag) * P(ending-hyph|tag)
            return 0;
        }
    }

    private class WittenBell extends SmoothScheme {
        int seen, unseen;

        public WittenBell(HashMap<String, Integer> wordFreq, HashMap<String, Integer> tagFreq, HashMap<String, Integer> wordTagFreq, HashMap<String, Integer> prevCurrTagFreq, int seen, int unseen) {
            super(wordFreq, tagFreq, wordTagFreq, prevCurrTagFreq);
            this.seen = seen;
            this.unseen = unseen;
        }

        // KIV
        public double getBigramTransition(String prevCurrTag) {
            //    (Seen tag)   : P(tag|prev-tag) = Count(prev-tag,tag) / [Count(prev-tag) + seen]
            // -> (Unseen tag) : P(tag|prev-tag) = seen / {unseen * [Count(prev-tag) + seen]}
            return 0;
        }

        public double getBigramEmission(String wordTag) {
            // TODO: -> (Unseen word) : P(word|tag) = seen / {unseen * [Count(tag) + seen]}
            return 0;
        }

        // KIV
        public double getUnknownWordBigramEmission(String wordTag, String prevWordTag) {
            // (Unseen tag) : P(word|tag) = seenTypes / {unseenTags * [Count(tag) + seenTags]}
            // (Unseen word) : P(unknown-word|tag) = P(unknown-word|tag) * P(capital-word|tag) * P(ending-hyph|tag)
            return 0;
        }
    }

    private List<String[]> getStrippedCorpus(List<String[]> taggedCorpus) {
        List<String[]> untaggedTestSent = new ArrayList<String[]>();
        for (String[] sentence : taggedCorpus) {
            String[] strippedSentence = new String[sentence.length];
            for (int index = 0; index < sentence.length; index++) {
                strippedSentence[index] = splitElement(sentence[index])[0];
            }
            untaggedTestSent.add(strippedSentence);
        }
        return untaggedTestSent;
    }

    private String[] splitElement(String element) {
        int index = element.lastIndexOf("/");
        String word = element.substring(0, index);
        String tag = element.substring(index, element.length()).replace("/", "");
        String[] splitString = new String[2];
        splitString[0] = word;
        splitString[1] = tag;
        return splitString;
    }
}