import math
import random ## added
trainRate = .8
class C45:

	"""Creates a decision tree with C4.5 algorithm"""
	def __init__(self, pathToData,pathToNames):
		self.filePathToData = pathToData
		self.filePathToNames = pathToNames
		self.data = []
		self.testdata = [] ## added
		self.alldata = []
		self.classes = []
		self.numAttributes = -1 
		self.attrValues = {}
		self.attributes = []
		self.tree = None

	def fetchData(self):
		with open(self.filePathToNames, "r") as file:
			classes = file.readline()
			self.classes = [x.strip() for x in classes.split(",")]
			#add attributes
			for line in file:
				[attribute, values] = [x.strip() for x in line.split(":")]
				values = [x.strip() for x in values.split(",")]
				self.attrValues[attribute] = values
		self.numAttributes = len(self.attrValues.keys())
		self.attributes = list(self.attrValues.keys())
		with open(self.filePathToData, "r") as file:
			for line in file:
				row = [x.strip() for x in line.split(",")]
				## Binary classification
				# if row[-1] != '0' and row[-1] != '1':
				#	 row[-1] = '1'
				if row != [] or row != [""]:
					self.alldata.append(row)
					if random.random() < trainRate:
						self.data.append(row)
					else:
						self.testdata.append(row)
		# self.printEverything()\

	def preprocessAllData(self):
		flag = 0
		for index,row in enumerate(self.alldata):
			if (self.classes).__contains__(row[0]):
				flag = 1
			if flag == 1:
				val = row[0]
				# print row[0]
				self.alldata[index].remove(row[0])
				self.alldata[index].append(val)
				flag = 0

	def preprocessTestData(self):
		flag = 0
		# print self.classes
		# print type(self.classes)
		for index,row in enumerate(self.testdata):
			if (self.classes).__contains__(row[0]):
				flag = 1
			if flag == 1:
				val = row[0]
				# print row[0]
				self.testdata[index].remove(row[0])
				self.testdata[index].append(val)
				flag = 0
		self.preprocessAllData()

	def preprocessData(self):
		for index,row in enumerate(self.data):
			for attr_index in range(self.numAttributes):
				if(not self.isAttrDiscrete(self.attributes[attr_index])):
					self.data[index][attr_index] = float(self.data[index][attr_index])
		flag = 0
		# print self.classes
		# print type(self.classes)
		for index,row in enumerate(self.data):
			if (self.classes).__contains__(row[0]):
				# print "LOL"
				flag = 1
				if flag == 1:
					data = row[0]
					# print row[0]
					self.data[index].remove(row[0])
					self.data[index].append(data)
					flag = 0
		# print self.data
		self.preprocessTestData()

		

	def printTree(self):
		self.printNode(self.tree)

	def printNode(self, node, indent=""):
		if not node.isLeaf:
			# print "Here"
			if node.threshold is None:
				#discrete
				# print ("DISCRETE")
				for index,child in enumerate(node.children):
					if child.isLeaf:  ## str(a[index]) was attributes[index]
						a = []
						a = self.attrValues.get(node.label)
						print(indent + node.label + " = " + str(a[index]) + " : " + child.label) 
					else:
						a = []
						a = self.attrValues.get(node.label)
						print(indent + node.label + " = " + str(a[index])  + " : ")
						self.printNode(child, indent + "	")
			else:
				#numerical
				# print ("NUMERICAL")
				leftChild = node.children[0]
				rightChild = node.children[1]
				if leftChild.isLeaf:
					print(indent + node.label + " <= " + str(node.threshold) + " : " + leftChild.label)
				else:
					print(indent + node.label + " <= " + str(node.threshold)+" : ")
					self.printNode(leftChild, indent + "	")

				if rightChild.isLeaf:
					print(indent + node.label + " > " + str(node.threshold) + " : " + rightChild.label)
				else:
					print(indent + node.label + " > " + str(node.threshold) + " : ")
					self.printNode(rightChild , indent + "	")


	def generateTree(self):
		self.tree = self.recursiveGenerateTree(self.data, self.attributes)

	def recursiveGenerateTree(self, curData, curAttributes):
		allSame = self.allSameClass(curData)
		if len(curData) == 0:
			## None found was Fail
			return Node(True, "Fail", None)
		elif allSame is not False:
			#return a node with that class
			return Node(True, allSame, None)
		elif len(curAttributes) == 0:
			#return a node with the majority class
			majClass = self.getMajClass(curData)
			return Node(True, majClass, None)
		else:
			(best,best_threshold,splitted) = self.splitAttribute(curData, curAttributes)
			remainingAttributes = curAttributes[:]
			remainingAttributes.remove(best)
			node = Node(False, best, best_threshold)
			node.children = [self.recursiveGenerateTree(subset, remainingAttributes) for subset in splitted]
			return node

	def getMajClass(self, curData):
		freq = [0]*len(self.classes)
		for row in curData:
			index = self.classes.index(row[-1])
			freq[index] += 1
		maxInd = freq.index(max(freq))
		return self.classes[maxInd]


	def allSameClass(self, data):
		if len(data) == 0:  ## added to avoid empy data error
			return False
		else:
			for row in data:
				if row[-1] != data[0][-1]:
					return False
			return data[0][-1]

	def isAttrDiscrete(self, attribute):
		if attribute not in self.attributes:
			raise ValueError("Attribute not listed")
		elif len(self.attrValues[attribute]) == 1 and self.attrValues[attribute][0] == "continuous":
			return False
		else:
			return True

	def gain(self,unionSet, subsets):
		#input : data and disjoint subsets of it
		#output : information gain
		S = len(unionSet)
		#calculate impurity before split
		impurityBeforeSplit = self.entropy(unionSet)
		#calculate impurity after split
		weights = [len(subset)/S for subset in subsets]
		impurityAfterSplit = 0
		for i in range(len(subsets)):
			impurityAfterSplit += weights[i]*self.entropy(subsets[i])
		#calculate total gain
		totalGain = impurityBeforeSplit - impurityAfterSplit
		return totalGain

	def splitAttribute(self, curData, curAttributes):
		splitted = []
		maxEnt = -1*float("inf")
		best_attribute = -1
		#None for discrete attributes, threshold value for continuous attributes
		best_threshold = None
		for attribute in curAttributes:
			indexOfAttribute = self.attributes.index(attribute)
			if self.isAttrDiscrete(attribute):
				#split curData into n-subsets, where n is the number of 
				#different values of attribute i. Choose the attribute with
				#the max gain
				valuesForAttribute = self.attrValues[attribute]
				subsets = [[] for a in valuesForAttribute]
				for row in curData:
					for index in range(len(valuesForAttribute)):
						if row[indexOfAttribute] == valuesForAttribute[index]:   ## indexOfAttribute was i
							subsets[index].append(row)
							break
				e = self.gain(curData, subsets) ## self.gain was gain
				if e > maxEnt:
					maxEnt = e
					splitted = subsets
					best_attribute = attribute
					best_threshold = None
			else:
				#sort the data according to the column.Then try all 
				#possible adjacent pairs. Choose the one that 
				#yields maximum gain
				curData.sort(key = lambda x: x[indexOfAttribute])
				for j in range(0, len(curData) - 1):
					if curData[j][indexOfAttribute] != curData[j+1][indexOfAttribute]:
						threshold = (curData[j][indexOfAttribute] + curData[j+1][indexOfAttribute]) / 2
						less = []
						greater = []
						for row in curData:
							if(row[indexOfAttribute] > threshold):
								greater.append(row)
							else:
								less.append(row)
						e = self.gain(curData, [less, greater])
						if e >= maxEnt:
							splitted = [less, greater]
							maxEnt = e
							best_attribute = attribute
							best_threshold = threshold
		return (best_attribute,best_threshold,splitted)

	def entropy(self, dataSet):
		S = len(dataSet)
		if S == 0:
			return 0
		num_classes = [0 for i in self.classes]
		for row in dataSet:
			classIndex = list(self.classes).index(row[-1])
			num_classes[classIndex] += 1
		num_classes = [x/S for x in num_classes]
		ent = 0
		for num in num_classes:
			ent += num*self.log(num)
		return ent*-1


	def log(self, x):
		if x == 0:
			return 0
		else:
			return math.log(x,2)

	def getNode(self):
		return self.tree
	
	def getTestData(self):
		return self.testdata

	def testNode(self,fnData = None):
		# print(self.attributes)
		success = 0
		if fnData is None:
			fnData = self.testdata
		for row in fnData:
			success += self.testRow(row)
		print "Success : ", success
		print "Total : ", len(fnData)
		print "Ratio : ", float(success)/len(fnData)

	def testDiscrete(self,row):
		node = self.tree
		for index,value in enumerate(row):
			# print "2"
			if not node.isLeaf:
				# print "3"
				if list(self.attrValues[node.label]).__contains__(value):
					# print "4"
					val = list(self.attrValues[node.label]).index(value)
					# print val,index
					# print node.label
					# print self.attributes[index]
					if node.label == self.attributes[index]:
						# print "5"
						node = node.children[val]
					else:
						# print "6"
						# print "Failure"
						return 0
				else:
					# print "7"
					continue
			else:
				# print "8"
				# print node.label
				if row[-1] == node.label:
					# print "9"
					# print "Success"
					return 1
				else:
					# print "10"
					# print "Failure"
					return 0
		return 0

	# def nodeRecurse(self,row,i,node):
	# 	attrIndex = val = list(self.attrValues[node.label]).index(i)
	# 	if node.isLeaf:
	# 		if row[-1] == node.label:
	# 			return 1
	# 		else:
	# 			return 0
	# 	elif node.label == self.attributes[i] and node.children.
	# 		node = node.children[attrIndex]
	# 	return 0

	# def testDiscrete1(self,row):
	# 	node = self.tree
	# 	return nodeRecurse(row,0,node)
	def printMyNODE(self,node,indent):
		if not node.isLeaf:
			print indent,"[", node.label, ",",node.threshold,",",len(node.children),"]"
			for child in node.children:
				# print "IN FOR"
				self.printMyNODE(child,"    " + indent)
		else:
			print indent,"[", node.label, ",",node.threshold,",",len(node.children),"]"



	def testNumerical(self,row,node):
		# self.printMyNODE(node,"")
		if not node.isLeaf:
			aI = self.attributes.index(node.label)
			# print node.label, node.threshold, row[aI]
			if row[aI] >= node.threshold:
				node = node.children[0]
				return self.testNumerical(row,node)
			else:
				node = node.children[1]
				return self.testNumerical(row,node)
		else:
			# print node.label
			if node.label == row[-1]:
				# print "Success"
				return 1
			else:
				# print "Failure"
				return 0



		
		return 0

	def testRow(self,row):
		node = self.tree
		# print row
		if node.threshold is None:
			#DISCRETE
			# print "1"
			return self.testDiscrete(row)

		else:
			# print "11"
			#NUMERICAL
			return self.testNumerical(row,node)
		# print "20"
		
		print "Failure"
		return 0
	
	def kCross(self,n):
		total = len(self.alldata)
		# print total
		testCount = float(total)/n
		# print testCount
		broken = []
		testSection = []
		trainingData = []
		for i in range(0,n):
			# print (i)*testCount , (i+1)*testCount
			start = int((i)*testCount)
			end = int((i+1)*testCount)
			# print len(self.alldata[start:end])
			broken.append(self.alldata[start:end])
		for i in range(0,n):
			trainingData = []
			testSection = broken[i]
			for j in range(0,n):
				if j != i:
					for row in broken[i]:
						trainingData.append(row)
			self.recursiveGenerateTree(trainingData,self.attributes)
			# print len(trainingData)
			# for row in testSection:
			self.testNode(testSection)
			print "\n"
			# print len(testSection)
		


class Node:
	def __init__(self,isLeaf, label, threshold):
		self.label = label
		self.threshold = threshold
		self.isLeaf = isLeaf
		self.children = []
