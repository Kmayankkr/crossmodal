
from os import listdir
from os.path import isfile, join
import random

def readTags(tagFilePath):
    tagfileNames = [f for f in listdir(tagFilePath) if isfile(join(tagFilePath, f))]

    tagImageIndices = []
    tagNames = []

    for filename in tagfileNames:
        tagImageIndices.append([int(line.rstrip('\n')) for line in open(tagFilePath+filename)])
        tagNames.append(filename.split("_")[0])

    return tagNames,tagImageIndices

def getRandomImageTagPairs(batchSize, tagNames, tagImageIndices):

    instancePerTag = batchSize/len(tagNames)

    if (instancePerTag<1):
        print "Warning!!! Number of Instances Per Tag is 0. Try to Increase BatchSize"

    residue = batchSize - instancePerTag * len(tagNames)

    randomImageTagPairs = []

    for i in xrange(0,len(tagImageIndices)):
        if (i==len(tagImageIndices)-1):
            indices = random.sample(range(0,len(tagImageIndices[i])), instancePerTag+residue)

            for j in indices:
                randomImageTagPairs.append([tagNames[i],tagImageIndices[i][j]])
        else:
            indices = random.sample(range(0,len(tagImageIndices[i])), instancePerTag)

            for j in indices:
                randomImageTagPairs.append([tagNames[i],tagImageIndices[i][j]])

    return randomImageTagPairs

def getWordEmbeddingsFromModel(filePath, tagNames):

    embeddings = []

    with open(filePath) as f:
        for line in f:
            terms = line.split(' ');

            if (terms[0].lower() in tagNames):
                embeddings.append([terms[0],[terms[i] for i in xrange(1,len(terms))]])

    return embeddings

def printEmbbeddingsToFile(wordEmbeddingsOutputFilePath, embeddings):
    f = open(wordEmbeddingsOutputFilePath, 'w')

    for i in xrange(0,len(embeddings)):
        f.write(str(embeddings[i][0])+" "+' '.join(embeddings[i][1]))

    f.close()

def getWordEmbeddingsFromOutputFile(wordEmbeddingsOutputFilePath):

    embeddings = dict()

    with open(wordEmbeddingsOutputFilePath) as f:
        for line in f:
            line.rstrip('\n')
            terms = line.split(' ')
            embeddings[terms[0]] = [float(terms[i]) for i in xrange(1,len(terms))]

    return embeddings

# embeddings_model = getWordEmbeddingsFromModel(wordEmbeddingFilePath, tagNames)
# embeddings_output = getWordEmbeddingsFromOutputFile(wordEmbeddingsOutputFilePath)

