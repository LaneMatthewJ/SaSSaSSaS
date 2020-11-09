import json, random, sys
import numpy as np
import getopt

def load_data(filePath):
    """
        This module specifically loads image data, 
        particularly RGB image data. 
    """
    f = open(filePath)
    wholeDataset = json.load(f)
    f.close()

    dataArray = np.array(wholeDataset['data']).astype('uint8')
    labelData = np.array(wholeDataset['labels']).astype('uint8')

    rgb = 3
    width = 80
    height = 80
    reshapedData = dataArray.reshape([-1, rgb, width, height])

    return (reshapedData, labelData)

def randomlySample(data, labels, numSamples): 
    totalSize = len(data)
    arangedData = np.arange(totalSize)
    permutedArangedData = np.random.permutation(arangedData)
    
    permutedData = data[permutedArangedData]
    permutedLabels = labels[permutedArangedData]

    sampledData = permutedData[:numSamples]
    sampledLabels = permutedLabels[:numSamples]
    return [sampledData, sampledLabels ]

def load_and_sample_data(fileName, numSamples=4000): 
    (reshapedData, labelData) = load_data(fileName)

    if numSamples > len(reshapedData): 
        raise ValueError(f"{numSamples} is greater than length of dataset: Length {len(reshapedData)}")

    sampledData = randomlySample(reshapedData, labelData, numSamples)
    return sampledData


def load_data_train_test_split(fileName, numSamples=4000): 
    [data, labels ] = load_and_sample_data(fileName, numSamples)

    # train test split: 70 % train, 15% test, 15% validation: 
    trainingSetLength = int(len(data) * 0.7)
    testingSetLength = trainingSetLength + int(len(data) * 0.15)
    validationSetLength = len(data)

    trainingSet = [ data[:trainingSetLength], labels[:trainingSetLength] ] 
    testingSet = [ data[trainingSetLength:testingSetLength ], labels[trainingSetLength:testingSetLength]]
    validationSet = [data[testingSetLength:], labels[testingSetLength:]]

    print("Training Set Data Length: ", len(trainingSet[0]), "  Label Length: ", len(trainingSet[1]))
    print("TestingSet Set Data Length: ", len(testingSet[0]), " Label Length: ", len(testingSet[1]))
    print("Validation Set Data Length: ", len(validationSet[0]), " Label Length: ", len(validationSet[1]))

    return (trainingSet, testingSet, validationSet)
def usage(): 
    print("""
        Usage: This is a python module used for  loading data for the shipsnet Kaggle Competition. , expected to be imported, and then have one of its functions used. 
        By default, from the `load_data` function, data are RGB values (i.e. 3 dimensional data) in an 80 X 80 matrix.
    
        Flattened Data Loading: 
        To load data in a 3 X 6400, use `load_data_flat(filePath)`.

        Sampling Data:         
        To load data sampled randomly, use `load_and_sample_data(filePath, optionalNumberOfSamples)`
        - In which optionalNumberOfSamples is any number 4000 or less, otherwise an error will be thrown. 

        Loading a Train/Test Split: 
        To load data in which there exists a train test split, use 'load_data_train_test_split(fileName, optionalNumberOfSamples)
        - In which optionalNumberOfSamples is any number 4000 or less, otherwise an error will be thrown. 
        - Data are permuted so as to mix true and false labels together using np.permute along an aranged 
               np.array for new indices for the training, testing, and validation sets. 

    """)

def main(): 
    try: 
        opts, args = getopt.getopt(sys.argv[1:],  "h", ["help"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(str(err))  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    
    output = None
    verbose = False
    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        else:
            assert(False, "unhandled option")

if __name__ == "__main__": 
    main()