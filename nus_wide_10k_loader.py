from os import listdir
from os.path import isfile, join
import random
from csv import reader
from random import shuffle

datasetFileListPath = "DatasetFilesList.txt" 
categoriesFilePath = "NUS-WIDE-10k-categories.txt" 
gloveEmbeddingsFilePath = "NUS_WIDE_10k_GloveEmbeddings.txt" 


# ['clouds', 'animal', 'flowers', 'food', 'grass', 'person', 'sky', 'toy', 'water', 'window']
categories = [] 	# list of categories for NUS_WIDE_10k. Used for creating one hot encoding.
fileList = [] 	#list of files in NUS_WIDE_10k dataset	
embedDict = {} 	#gloVe embeddings dictionary
train_fileList = []
test_fileList = []

category_wise_instances = {}
source_categories = [0,1,2,3,4,5]
target_categories = [6,7,8,9,4,5]

source_train_instances = []
source_test_instances = []
target_train_instances = []
target_test_instances = []

# loads list of all instances in fileList, training instances in train_fileList, test instances in test_fileList
def setup_batch(base_path, source_train_test_ratio, target_train_test_ratio):
	global categories,fileList,embedDict,datasetFileListPath, categoriesFilePath, gloveEmbeddingsFilePath, train_fileList, test_fileList
	global source_train_instances, source_test_instances, target_train_instances, target_test_instances
	
	datasetFileListPath = base_path + datasetFileListPath
	categoriesFilePath = base_path + categoriesFilePath
	gloveEmbeddingsFilePath = base_path + gloveEmbeddingsFilePath

	categories = load_file(categoriesFilePath)	

	fileList = load_file(datasetFileListPath)
	shuffle(fileList)

	# get UniqueTags
	allTags = [ ((x.split("/")[1]).split("_")[0]) for x in fileList ]
	uTags = set(allTags)
	uniqueTags = list(uTags)
	#print "unique Tags", len(uniqueTags)

	# get glove embeddings
	embeddings = get_word_embeddings_from_model(gloveEmbeddingsFilePath,uniqueTags) 

	for x in embeddings:
		embedDict[x[0]] = x[1] 

	for x in fileList:

		cat_key = categories.index(x.split("/")[0])

		if cat_key in category_wise_instances:
			category_wise_instances[cat_key].append([embedDict[((x.split("/")[1]).split("_")[0])], x, cat_key])
		else:
			category_wise_instances[cat_key] = [[embedDict[((x.split("/")[1]).split("_")[0])], x, cat_key]]

	for cat_key in source_categories:
		total_instances = len(category_wise_instances[cat_key])
		num_train_instances = int(source_train_test_ratio * total_instances)
		num_test_instances = total_instances - num_train_instances
		
		for i in range(num_train_instances):
			source_train_instances.append(category_wise_instances[cat_key][i])

		for i in range(num_test_instances):
			source_test_instances.append(category_wise_instances[cat_key][num_train_instances+i])

	for cat_key in target_categories:
		total_instances = len(category_wise_instances[cat_key])
		num_train_instances = int(target_train_test_ratio * total_instances)
		num_test_instances = total_instances - num_train_instances
		
		for i in range(num_train_instances):
			target_train_instances.append(category_wise_instances[cat_key][i])

		for i in range(num_test_instances):
			target_test_instances.append(category_wise_instances[cat_key][num_train_instances+i])

	# train_fileList = fileList[0:num_train_instances]
	# test_fileList = fileList[num_train_instances:]

def print_vertical(xx):
	for x in xx:
		print x 
	
# read data from file    
def load_file(filename):
	
	val = [] 
	with open(filename) as afile:
		r = reader(afile)
		for line in r:
			val.append(line[0]) 
	
	return val 

# Use this to get a random batch of source train instances	
def get_batch_source_train(batchSize):
	return get_batch_from_instance_list(batchSize, source_train_instances)

# Use this to get a random batch of source test instances	
def get_batch_source_test(batchSize):
	return get_batch_from_instance_list(batchSize, source_test_instances) 

# Use this to get a random batch of target train instances
def get_batch_target_train(batchSize):
	return get_batch_from_instance_list(batchSize, target_train_instances)

# Use this to get a random batch of target test instances
def get_batch_target_test(batchSize):
	return get_batch_from_instance_list(batchSize, target_test_instances)

# Use this to get a random batch from an instance list
def get_batch_from_instance_list(batch_size, instance_list):
	
	randomFiles = random.sample(range(0,len(instance_list)), batch_size)
	batchInstances = [ instance_list[i] for i in randomFiles] 
	
	# contains array of [tag,imagePath,oneHotCategory]
	ret=[] 
	
	for x in batchInstances:
		zeroList=[0]*len(categories) 
		zeroList[x[2]] = 1 
		x[2] = zeroList 
		ret.append(x)
	
	return ret

# Load word embeddings from Glove Model for all words in tagNames
def get_word_embeddings_from_model(filePath, tagNames):

	embeddings = []

	with open(filePath) as f:
		for line in f:
			terms = line.split(' ') 

			if (terms[0].lower() in tagNames):
				
				embeddings.append([terms[0],[terms[i] for i in xrange(1,len(terms))]])
			
	return embeddings

# print all embeddings supplied to a txt file
def print_embbeddings_to_file(wordEmbeddingsOutputFilePath, embeddings):
	f = open(wordEmbeddingsOutputFilePath, 'w')

	for i in xrange(0,len(embeddings)):
		f.write(str(embeddings[i][0])+" "+' '.join(embeddings[i][1]))

	f.close()
