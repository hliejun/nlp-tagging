/**
 * Model encapsulates the state of the POS model with
 * the methods that the model can be utilised for.
 * This class is serialisable.
 *
 * @author Huang Lie Jun (A0123994W)
 * @version 1.0
 * @since 2017-10-08
 */
import java.io.*;
import java.util.*;

/**
 * Enumerated types for labelling smoothing techniques
 */
enum Technique {LAPLACE, WITTENBELL, KNESERNEY}
enum Type {WORD, TAG, BOTH}

public class Model implements Serializable {
    private Technique smoothingMode = Technique.LAPLACE;
    private HashMap<String, Integer> wordFreq, tagFreq, wordTagFreq, prevCurrTagFreq;
    private HashMap<String, Float> transitionProbMatrix, emissionProbMatrix;
    private List<String> uniqueWords, uniqueTags;
    private List<List<String>> results;
    private String startTag, equate, separator, entrySeparator, keyValueSeparator, segmentSeparator;

    public Model() {
        super();
        initConstants();
    }

    /**
     * This method constructs the model frequency tables
     * from the training corpus and calculates the
     * transition and emission probabilities of seen words
     * and tags.
     *
     * @param trainingCorpus
     */
    public void train(List<String[]> trainingCorpus) {
        indexCorpus(trainingCorpus);
        buildTransitionMatrix();
        buildEmissionMatrix();
    }

    /**
     * This method applies the Viterbi algorithm on a
     * test corpus, and applies the smoothing scheme
     * on unknown words. If the test corpus is tagged,
     * it will assess the accuracy of tagging and return
     * the accuracy. Otherwise, it will return 0.
     *
     * @param testCorpus
     * @param smoothingScheme
     * @param isTagged
     *
     * @return float
     */
    public float test(List<String[]> testCorpus, Technique smoothingScheme, boolean isTagged) {
        Set<String> testWords, seenWords, unseenWords;
        HashMap<String, Integer> testWordsFreq;
        int correct = 0, total = 0;
        SmoothScheme smoother = null;
        List<String[]> untaggedTestCorpus = isTagged ? getStrippedCorpus(testCorpus) : testCorpus;
        switch (smoothingScheme) {
            case LAPLACE:
                smoother = new Laplace(wordFreq, tagFreq, wordTagFreq, prevCurrTagFreq, 1);
                break;
            case WITTENBELL:
                testWordsFreq = new HashMap<String, Integer>();
                for (String[] sentence : untaggedTestCorpus) {
                    for (String word : sentence) {
                        incrementFreqTable(testWordsFreq, word);
                    }
                }
                seenWords = wordFreq.keySet();
                unseenWords = new HashSet<String>(testWordsFreq.keySet());
                unseenWords.removeAll(seenWords);
                smoother = new WittenBell(wordFreq, tagFreq, wordTagFreq, prevCurrTagFreq, seenWords.size(), unseenWords.size());
                break;
            default:
                smoother = new Laplace(wordFreq, tagFreq, wordTagFreq, prevCurrTagFreq, 1);
                break;
        }
        results = new ArrayList<List<String>>();
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
                    Float alpha, beta;
                    String currentTag = uniqueTags.get(tagIndex);
                    if (currentTag == startTag) {
                        continue;
                    } else if (wordIndex == 0) {
                        alpha = transitionProbMatrix.get(startTag + separator + currentTag);
                        alpha = (alpha != null) ? alpha : 0.0f;
                        beta = emissionProbMatrix.get(currentWord + separator + currentTag);
                        beta = (beta != null) ? beta : (countWord(currentWord) == 0)
                            ? smoother.getBigramEmission(currentWord, currentTag)
                            : 0.0f;
                        pathProbMatrix[tagIndex][wordIndex] = alpha * beta;
                        backpointerMatrix[tagIndex][wordIndex] = -1;
                    } else {
                        bestPrevTagIndex = 0;
                        maxPathValue = 0.0f;
                        for (int prevTagIndex = 0; prevTagIndex < uniqueTags.size(); prevTagIndex++) {
                            String prevTag = uniqueTags.get(prevTagIndex);
                            alpha = transitionProbMatrix.get(prevTag + separator + currentTag);
                            alpha = (alpha != null) ? alpha : 0.0f;
                            double value = pathProbMatrix[prevTagIndex][wordIndex - 1] * alpha;
                            if (value >= maxPathValue) {
                                maxPathValue = value;
                                bestPrevTagIndex = prevTagIndex;
                            }
                        }
                        beta = emissionProbMatrix.get(currentWord + separator + currentTag);
                        beta = (beta != null) ? beta : (countWord(currentWord) == 0)
                                ? smoother.getBigramEmission(currentWord, currentTag)
                                : 0.0f;
                        pathProbMatrix[tagIndex][wordIndex] = maxPathValue * beta;
                        backpointerMatrix[tagIndex][wordIndex] = bestPrevTagIndex;
                    }
                }
            }
            int bestEndIndex = 0;
            maxPathValue = 0.0f;
            for (int tagIndex = 0; tagIndex < uniqueTags.size(); tagIndex++) {
                String tag = uniqueTags.get(tagIndex);
                if (tag == startTag) {
                    continue;
                }
                double pathValue = pathProbMatrix[tagIndex][sentence.length - 1];
                if (pathValue >= maxPathValue) {
                    maxPathValue = pathValue;
                    bestEndIndex = tagIndex;
                }
            }
            pathProbMatrix[uniqueTags.size()][sentence.length - 1] = maxPathValue;
            backpointerMatrix[uniqueTags.size()][sentence.length - 1] = bestEndIndex;
            int prevStateIndex = bestEndIndex;
            int prevSequenceIndex = sentence.length - 1;
            List<String> prediction = new ArrayList<String>();
            while (prevStateIndex != -1 && prevSequenceIndex >= 0) {
                String tag = uniqueTags.get(prevStateIndex);
                String word = sentence[prevSequenceIndex];
                prediction.add(0, word + separator + tag);
                prevStateIndex = backpointerMatrix[prevStateIndex][prevSequenceIndex];
                prevSequenceIndex -= 1;
            }
            if (isTagged) {
                for (int predictionIndex = 0; predictionIndex < prediction.size(); predictionIndex++) {
                    if (prediction.get(predictionIndex).equals(taggedSentence[predictionIndex])) {
                        correct += 1;
                    }
                    total += 1;
                }
            }
            results.add(prediction);
        }
        return isTagged ? ((float)correct / total) : 0.0f;
    }

    /**
     * This method will run tests on the test corpus
     * with different smoothing techniques and select
     * the optimal technique for actual tagging.
     *
     * @param testCorpus
     */
    public void tune(List<String[]> testCorpus) {
        Technique[] techniques = new Technique[]{Technique.LAPLACE, Technique.WITTENBELL};
        float currentAccuracy = 0, bestAccuracy = 0;
        for (Technique technique : techniques) {
            currentAccuracy = this.test(testCorpus, technique, true);
            if (currentAccuracy >= bestAccuracy) {
                bestAccuracy = currentAccuracy;
                this.smoothingMode = technique;
            }
        }
    }

    /**
     * This method will perform n-fold cross-validation
     * on the corpus provided. n segments will be partitioned
     * and prior to validating every segment, training will
     * be conducted on the remaining sentences in the corpus.
     * The accuracies obtained will be averaged and returned.
     *
     * @param corpus
     * @param n
     *
     * @return float
     */
    public float crossValidate(List<String[]> corpus, int n) {
        float averageAccuracy = 0;
        if (n <= 0) {
            System.err.println("Cross validation fold must be positive.");
            return 0;
        } else {
            List<String[]> validationCorpus, trainingCorpus, remCorpus;
            int intervalSize = (int)Math.ceil((float)corpus.size() / n);
            Model cvModel = new Model();
            for (int start = 0; start < corpus.size(); start += intervalSize) {
                int end = start + intervalSize;
                end = (end <= corpus.size()) ? end : corpus.size();
                validationCorpus = new ArrayList<String[]>(corpus.subList(start, end));
                trainingCorpus = (start == 0)
                        ? new ArrayList<String[]>(corpus.subList(end, corpus.size()))
                        : new ArrayList<String[]>(corpus.subList(0, start));
                if (end < corpus.size()) {
                    remCorpus = new ArrayList<String[]>(corpus.subList(end, corpus.size()));
                    trainingCorpus.addAll(remCorpus);
                }
                cvModel.train(trainingCorpus);
                float accuracy = cvModel.test(validationCorpus, this.getBestTechnique(), true);
                averageAccuracy += accuracy;
            }
            return averageAccuracy / n;
        }
    }

    /**
     * This method applies actual tagging on the given untagged
     * corpus, using the trained and tuned paramters of the model.
     * The tagged result will be returned.
     *
     * @param corpus
     *
     * @return List
     */
    public List<List<String>> tag(List<String[]> corpus) {
        test(corpus, smoothingMode, false);
        return results;
    }

    /*** Accessors ***/

    public Technique getBestTechnique() {
        return this.smoothingMode;
    }

    public HashMap<String, Float> getTransitionProbMatrix() {
        return this.transitionProbMatrix;
    }

    public HashMap<String, Float> getEmissionProbMatrix() {
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

    /**
     * This interface specifies the required common
     * methods across all smoothing techniques.
     */
    private interface Smoothing {
        public float getBigramTransition(String prevTag, String currTag);
        public float getBigramEmission(String word, String tag);
    }

    /**
     * This abstract class is the parent class of all
     * smoothing techniques. It constructs each smoothing
     * scheme with the required frequency tables provided
     * by the model.
     */
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

    /**
     * The Laplace class implements the smoothing methods
     * with the Laplace smoothing technique, given the Laplace factor.
     */
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

        public float getBigramTransition(String prevTag, String currTag) {
            return ((float)countPrevCurrTag(prevTag, currTag) + 1) / ((float)countTag(prevTag) + ((float)laplaceFactor * tagFreq.size()));
        }

        public float getBigramEmission(String word, String tag) {
            return ((float)countWordTag(word, tag) + 1) / ((float)countTag(tag) + ((float)laplaceFactor * tagFreq.size()));
        }
    }

    /**
     * The WittenBell class implements the smoothing methods
     * with the Witten Bell smoothing technique, given seen and unseen
     * word count.
     */
    private class WittenBell extends SmoothScheme {
        int seen, unseen;

        public WittenBell(HashMap<String, Integer> wordFreq, HashMap<String, Integer> tagFreq, HashMap<String, Integer> wordTagFreq, HashMap<String, Integer> prevCurrTagFreq, int seen, int unseen) {
            super(wordFreq, tagFreq, wordTagFreq, prevCurrTagFreq);
            this.seen = seen;
            this.unseen = unseen;
        }

        public float getBigramTransition(String prevTag, String currTag) {
            return (float)seen / ((float)unseen * ((float)countTag(prevTag) + (float)seen));
        }

        public float getBigramEmission(String word, String tag) {
            return (float)seen / ((float)unseen * ((float)countTag(tag) + (float)seen));
        }
    }

    // Kneser Ney Smoothing is shelved due to time constraint
    private class KneserNey extends SmoothScheme {

        public KneserNey(HashMap<String, Integer> wordFreq, HashMap<String, Integer> tagFreq, HashMap<String, Integer> wordTagFreq, HashMap<String, Integer> prevCurrTagFreq) {
            super(wordFreq, tagFreq, wordTagFreq, prevCurrTagFreq);
        }

        public float getBigramTransition(String prevTag, String currTag) {
            // TODO: Transition Probability: P(tag|prev-tag) = alpha(prev-tag) x prev-count(prev-tag, tag) / sum(prev-count(prev-tag, all seen tags))
            // alpha(prev-tag) = [1 - sum(discounted_probability(all seen tags|prev-tag)] / [1 - sum(discounted_probability(all seen tags)]
            return 0;
        }

        public float getBigramEmission(String word, String tag) {
            // TODO: Emission Probability: P(word|tag) = alpha(tag) x prev-count(tag, word) / sum(prev-count(tag, all seen words))
            // alpha(prev-tag) = [1 - sum(discounted_probability(all seen words|tag)] / [1 - sum(discounted_probability(all seen words)]
            return 0;
        }
    }

    /**
     * This method creates the word and tag frequency tables
     * based on the provided corpus.
     *
     * @param corpus
     */
    private void indexCorpus(List<String[]> corpus) {
        String prevWord, currWord, prevTag, currTag, prevCurrTag;
        String[] currWordTag;
        String prev = "", curr = "";
        wordFreq = new HashMap<String, Integer>();
        tagFreq = new HashMap<String, Integer>();
        wordTagFreq = new HashMap<String, Integer>();
        prevCurrTagFreq = new HashMap<String, Integer>();
        for (String[] sentence : corpus) {
            for (int index = 0; index < sentence.length; index++) {
                curr = sentence[index];
                currWordTag = splitElement(curr);
                currWord = currWordTag[0];
                currTag = currWordTag[1];
                incrementFreqTable(wordFreq, currWord);
                incrementFreqTable(tagFreq, currTag);
                incrementFreqTable(wordTagFreq, curr);
                if (index == 0) {
                    incrementFreqTable(tagFreq, startTag);
                    prevTag = startTag;
                } else {
                    prev = sentence[index - 1];
                    prevTag = splitElement(prev)[1];
                }
                prevCurrTag = prevTag + separator + currTag;
                incrementFreqTable(prevCurrTagFreq, prevCurrTag);
            }
        }
        uniqueWords = new ArrayList<String>(wordFreq.keySet());
        Collections.sort(uniqueWords);
        uniqueTags = new ArrayList<String>(tagFreq.keySet());
        Collections.sort(uniqueTags);
    }

    /**
     * This method creates the transition probability matrix
     * prior to any testing or tuning.
     */
    private void buildTransitionMatrix() {
        String prevTag, currTag, prevCurrTag;
        transitionProbMatrix = new HashMap<String, Float>();
        for (int row = 0; row < uniqueTags.size(); row++) {
            currTag = uniqueTags.get(row);
            for (int col = 0; col < uniqueTags.size(); col++) {
                prevTag = uniqueTags.get(col);
                prevCurrTag = prevTag + separator + currTag;
                float probability = (float)countPrevCurrTag(prevTag, currTag) / countTag(prevTag);
                if (probability > 0) {
                    transitionProbMatrix.put(prevCurrTag, probability);
                }
            }
        }
    }

    /**
     * This method creates the emission probability matrix
     * prior to any testing or tuning.
     */
    private void buildEmissionMatrix() {
        String currWord, currTag, wordTag;
        emissionProbMatrix = new HashMap<String, Float>();
        for (int row = 0; row < uniqueWords.size(); row++) {
            currWord = uniqueWords.get(row);
            for (int col = 0; col < uniqueTags.size(); col++) {
                currTag = uniqueTags.get(col);
                wordTag = currWord + separator + currTag;
                float probability = (float)countWordTag(currWord, currTag) / countTag(currTag);
                if (probability > 0) {
                    emissionProbMatrix.put(wordTag, probability);
                }
            }
        }
    }

    /**
     * This helper method removes the tags from a tagged corpus
     * for testing and rating purposes. It returns the untagged
     * version of the corpus.
     *
     * @param taggedCorpus
     *
     * @return List
     */
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

    /**
     * This method splits the word-tag string into its separate
     * entities.
     *
     * @param element
     *
     * @return String[]
     */
    private String[] splitElement(String element) {
        int index = element.lastIndexOf(separator);
        String word = element.substring(0, index);
        String tag = element.substring(index, element.length()).replace(separator, "");
        String[] splitString = new String[]{word, tag};
        return splitString;
    }

    /*** Counters and Incrementers ***/

    private int countWord(String word) {
        Integer wordCount = wordFreq.get(word);
        return (wordCount != null) ? (int)wordCount : 0;
    }

    private int countTag(String tag) {
        Integer tagCount = tagFreq.get(tag);
        return (tagCount != null) ? (int)tagCount : 0;
    }

    private int countWordTag(String word, String tag) {
        String wordTag = word + separator + tag;
        Integer wordTagCount = wordTagFreq.get(wordTag);
        return (wordTagCount != null) ? (int)wordTagCount : 0;
    }

    private int countPrevCurrTag(String prevTag, String currTag) {
        String prevCurrTag = prevTag + separator + currTag;
        Integer prevCurrTagCount = prevCurrTagFreq.get(prevCurrTag);
        return (prevCurrTagCount != null) ? (int)prevCurrTagCount : 0;
    }

    private void incrementFreqTable(HashMap<String, Integer> table, String key) {
        Integer value = table.get(key);
        value = (value != null) ? value + 1 : 1;
        table.put(key, value);
    }

    private void initConstants() {
        startTag = "<s>";
        equate = "====";
        separator = "/";
        entrySeparator = "@@@@";
        keyValueSeparator = "::::";
        segmentSeparator = "\n\n\n\n";
    }

    /*** Serializable Methods ***/
    private void writeObject(ObjectOutputStream serializer) throws IOException {
        serializer.writeObject(smoothingMode);
        serializer.writeObject(wordFreq);
        serializer.writeObject(tagFreq);
        serializer.writeObject(wordTagFreq);
        serializer.writeObject(prevCurrTagFreq);
        serializer.writeObject(transitionProbMatrix);
        serializer.writeObject(emissionProbMatrix);
        serializer.writeObject(uniqueWords);
        serializer.writeObject(uniqueTags);
    }

    @SuppressWarnings("unchecked")
    private void readObject(ObjectInputStream deserializer) throws IOException, ClassNotFoundException {
        smoothingMode = (Technique) deserializer.readObject();
        wordFreq = (HashMap<String, Integer>) deserializer.readObject();
        tagFreq = (HashMap<String, Integer>) deserializer.readObject();
        wordTagFreq = (HashMap<String, Integer>) deserializer.readObject();
        prevCurrTagFreq = (HashMap<String, Integer>) deserializer.readObject();
        transitionProbMatrix = (HashMap<String, Float>) deserializer.readObject();
        emissionProbMatrix = (HashMap<String, Float>) deserializer.readObject();
        uniqueWords = (List<String>) deserializer.readObject();
        uniqueTags = (List<String>) deserializer.readObject();
        initConstants();
    }
}