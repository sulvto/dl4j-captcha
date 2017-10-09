package me.qinchao.captcha;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Created by sulvto on 17-10-5.
 */
public class TrainAndSaveModel {

    private static Logger log = LoggerFactory.getLogger(TrainAndSaveModel.class);

    public static void main(String[] args) throws Exception {
        // image information
        // 80 * 30 grayscale
        // grayscale implies single channel
        int channels = 1;
        int rngseed = 123;
        Random randNumGen = new Random(rngseed);
        int batchSize = 128;
        // 40 + 26 * 2 * 4
        int outputNum = 248;
        int numEpochs = 15;


        // Define the File Paths
        File trainData = new File(Const.TRAINING_PATH);
        // File testData = new File(DATA_PATH + "/mnist_png/testing");


        // Define the FileSplit(PATH, ALLOWED FORMATS,random)

        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        // FileSplit test = new FileSplit(testData,NativeImageLoader.ALLOWED_FORMATS,randNumGen);

        // Extract the parent path as the image label

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        ImageRecordReader recordReader = new ImageRecordReader(CaptchaUtils.HEIGHT, CaptchaUtils.WIDTH, channels, labelMaker);

        // Initialize the record reader
        // add a listener, to extract the name

        recordReader.initialize(train);
//        recordReader.setListeners(new LogRecordListener());

        // DataSet Iterator

        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);

        // Scale pixel values to 0-1

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);


        // Build Our Neural Network

        log.info("**** Build Model ****");
        // learning rate schedule in the form of <Iteration #, Learning Rate>
        Map<Integer, Double> lrSchedule = new HashMap<Integer, Double>();
        lrSchedule.put(0, 0.01);
        lrSchedule.put(1000, 0.005);
        lrSchedule.put(3000, 0.001);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngseed)
                .iterations(1) // Training iterations as above
                .regularization(true).l2(0.0005)
                /*
                    Uncomment the following for learning decay and bias
                 */
                .learningRate(.01)//.biasLearningRate(0.02)
                /*
                    Alternatively, you can use a learning rate schedule.

                    NOTE: this LR schedule defined here overrides the rate set in .learningRate(). Also,
                    if you're using the Transfer Learning API, this same override will carry over to
                    your new model configuration.
                */
                .learningRateDecayPolicy(LearningRatePolicy.Schedule)
                .learningRateSchedule(lrSchedule)
                /*
                    Below is an example of using inverse policy rate decay for learning rate
                */
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse)
                //.lrPolicyDecayRate(0.001)
                //.lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS) //To configure: .updater(new Nesterovs(0.9))
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        //Note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(CaptchaUtils.HEIGHT, CaptchaUtils.WIDTH, 1)) //See note below
                .backprop(true).pretrain(false).build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();


        // 初始化用户界面后端
        UIServer uiServer = UIServer.getInstance();

        // 设置网络信息（随时间变化的梯度、分值等）的存储位置。这里将其存储于内存。
        StatsStorage statsStorage = new InMemoryStatsStorage();         // 或者： new FileStatsStorage(File)，用于后续的保存和载入

        // 将StatsStorage实例连接至用户界面，让StatsStorage的内容能够被可视化
        uiServer.attach(statsStorage);

        //  然后添加StatsListener来在网络定型时收集这些信息
        model.setListeners(new StatsListener(statsStorage));
//        model.setListeners(new ScoreIterationListener(10));

        log.info("***** TRAIN MODEL ********");
        for (int i = 0; i < numEpochs; i++) {
            model.fit(dataIter);
        }

        log.info("****** SAVE TRAINED MODEL ******");

        // Where to save model
        File locationToSave = new File(Const.SAVE_MODEL_FILE);

        // boolean save Updater
        boolean saveUpdater = false;

        // ModelSerializer needs modelname, saveUpdater, Location

        ModelSerializer.writeModel(model, locationToSave, saveUpdater);

        log.info("done. ");
    }

}
