import networkx as nx
import math
import matplotlib.pyplot as plt

def pest(outputs, k):
	'''Calculates the probability estimate, p, that a given data point in the region defined
	by the values in outputs will be in class k
	INPUT:
		outputs - 1D collection of integer values, presumably the outcome class labels for some region 
		of input space
		k -  an integer corresponding to a class label
	OUTPUT:
		p - a float estimate of the probability for class k
	'''
	nm = len(outputs) #number of data values in this region
	p = 0.0
	for o in outputs:
		if o == k: p += 1
	return p/nm
	
def compute_class(data_row_indicies, outputs, clazz):
	k = None
	max_p = -1
	out = [outputs[index] for index in data_row_indicies]
	for c in clazz:
		temp = pest(out, c)
		if temp > max_p:
			max_p = temp
			k = c
	return k
	
	
def __assign_class(tree, node, clazz, outputs):
	data_row_indicies = tree.node[node]['data']
	return compute_class(data_row_indicies, outputs, clazz) 
	
def cross_entropy(outputs, clazz):
	'''Calculates the cross entropy for the region defined by the values in outputs
	INPUT:
		outputs - 1D collection of integer values, presumably the outcome class labels for some region 
		of input space
		clazz - 1D collection of integers corresponding to the class labels
	OUTPUT:
		ce - a float estimate of the cross entropy
	'''
	def ent(k):
		p = pest(outputs,k)
		if p==0: return 0
		else: return p * math.log(p)
	return -1.0 * reduce(lambda accum, k: accum+ ent(k), clazz, 0) 
	
def gini_index(outputs, clazz):
	def gini(k):
		p = pest(outputs,k)
		return p - p*p
	return reduce(lambda accum, k: accum + gini(k), clazz, 0)
	
def __initiate_node(tree, node):
	tree.add_node(node)
	keys = ['j','i','s','data']
	for k in keys: tree.node[node][k] = None
		
def __grow_path(tree, coords, input, output, clazz, split_func, max_rm, meta, data_row_indicies, parent_node = None, weight = 0):
	'''INPUT:
	tree - current digraph
	node - parent node
	input - input data
	output - output data set
	clazz - integer class labels
	split_func - splitting function Qm
	max_rm - stopping criteria
	meta - input space labels
	'''
	
	if len(data_row_indicies)<=max_rm: #add a terminal node
		k = compute_class(data_row_indicies, output, clazz)
		label = '{0}, C={1}'.format(coords,k)
		__initiate_node(tree, label)
		tree.node[label]['c'] = coords
		tree.node[label]['o'] = k
		if parent_node: tree.add_weighted_edges_from([(parent_node, label, weight)])
		return #this path is complete
	temp1 = 0
	temp2 = 0
	min_balance = float("inf")
	min_split = float("inf")
	split_j = None
	split_i = None
	for idx in data_row_indicies: #loop over data
		out1 = []
		out2 = []	
		for col in range(input.shape[1]): #loop over feature space
			split_value = input[idx][col]
			out1 = [output[index] for index in data_row_indicies if input[index][col] <= split_value]
			out2 = [output[index] for index in data_row_indicies if input[index][col] > split_value]
			if len(out1)>0 and len(out2)>0:
				temp1 = split_func(out1, clazz) + split_func(out2, clazz)
				temp2 = math.fabs(len(out1)-len(out2))
				if temp1 <= min_split and temp2 <= min_balance:
					split_j = col
					split_i = idx
					min_split = temp1
					min_balance = temp2
	
	#create the node for this split
	split_value = input[split_i][split_j]
	label = '{0}<={1},{2}'.format(meta[split_j],split_value, coords)
	tree.add_node(label)
	tree.node[label]['j'] = split_j
	tree.node[label]['i'] = split_i
	tree.node[label]['s'] = split_value
	tree.node[label]['c'] = coords
	tree.node[label]['data'] = data_row_indicies
	tree.node[label]['o'] = __assign_class(tree, label, clazz, output)
	if parent_node: tree.add_weighted_edges_from([(parent_node, label, weight)])
	
	#grow paths for split
	left_data = [index for index in data_row_indicies if input[index][split_j] <= split_value]
	right_data = [index for index in data_row_indicies if input[index][split_j] > split_value]
	l,k = [int(x) for x in coords.split(',')] #level and column for this node
	left_coords = '{0},{1}'.format(str(l+1),str(2*k))
	right_coords = '{0},{1}'.format(str(l+1),str(2*k+1))
	__grow_path(tree, left_coords, input, output, clazz, split_func, max_rm, meta, left_data, label, 0)
	__grow_path(tree, right_coords, input, output, clazz, split_func, max_rm, meta, right_data, label, 1)		

def build_tree(input, output, clazz, meta, split_func=cross_entropy, max_rm=5):
	'''Computes a classification decision tree given the training data and splitting function
	INPUT:
		input - a numpy array of the input data, each row should correspond to a single input 
			vector, i.e. columns represent a feature or dimension of the input space.  
		output - the 1D output values corresponding to the input data
		clazz - 1D collection of integers corresponding to the class labels
		meta - 1D collection of labels for feature space
		split_func - function used as splitting criteria
		max_rm - the maximum number of input data points allowed to be associated with a 
			terminal node. This is used as the stopping criteria for growing the tree
	OUTPUT:
		tree - A networkx DiGraph (binary, exactly two edges from each node except for terminal 
		nodes, no loops). 
		Nodes are named with the convention N,M where N is the row level and M is the node count
		Each node is assigned:
			j - the column number of the feature to be split on (0 based).
			i - the index of output to split on
			s - the value to split on (output[i], provided for convenience)
			o - the class assigned to inputs in this region
			data - a list of the row indicies from the training input that are in the region defined
				by the node
			c - the coordinates for this node
		Each edge from a given node is assigned a weight of 0 or 1. Edges with 0 weight 
		define the path to the next node for which values the j feature value in "data"
		is less or equal to s and those with weight 1 are for those greater than s. 	
	'''
	tree = nx.DiGraph()
	__grow_path(tree, '0,0', input, output, clazz, split_func, max_rm, meta, range(len(output)))
	return tree
	
def draw_tree(tree, min_delta=100, node_size=500):
	pos = {}
	max_level = 0
	for n in tree.nodes():
		temp = int(tree.node[n]['c'].split(',')[0])
		if temp > max_level: max_level = temp
	
	for node in tree.nodes():
		l,k = [int(x) for x in tree.node[node]['c'].split(',')]
		ck = (2**l - 1) / 2.0
		delta = 2**(max_level - l) * min_delta
		pos[node]=((k-ck)*delta, (max_level-l)*min_delta)
	
	nx.draw_networkx_nodes(tree,pos,node_size=node_size)
	nx.draw_networkx_edges(tree,pos,alpha=0.5,width=2)
	nx.draw_networkx_labels(tree,pos)
	plt.axis('off')
	return plt
	
def find_node_by_coords(tree, coords):
	for n in tree.nodes():
		if tree.node[n]['c']==coords: return n
	
def decide(tree, input, coords='0,0'):
	node = find_node_by_coords(tree, coords)
	if tree.node[node]['s']:
		col = tree.node[node]['j']
		l,k = [int(x) for x in coords.split(',')]
		split_val = tree.node[node]['s']
		v = input[col]
		if v <= split_val: return decide(tree, input, '{0},{1}'.format(str(l+1),str(2*k)))
		else: return decide(tree, input, '{0},{1}'.format(str(l+1),str(2*k + 1)))
	else: return tree.node[node]['o'] 