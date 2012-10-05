import networkx as nx

def forward_prop(DG, clear_current = True):
	'''Computes the feed-forward network output for every node in the given a directed graph DG.
	   INPUT:
	     DG - a directed graph
	       The nodes of DG must contain the following attributes:
	         af : the node's activation function 
	              if the node is a bias node or an input node, this attribute should just be a constant value
	          o : the output attribute of the node, should be set to None. if clear_current is set to True this
	              attribute will be added automatically
	       The edges should be weighted edges, with the following attributes:
	         w  : weight
	    clear_current - if True, the algorithm will first traverse DG to clear each node's 'o'
	                    attribute. This is useful for algorithms requiring multiple propagations through 
	                    DG
	  OUTPUT:
	    Every node in DG will have the attribute 'o' added representing that node's output value    
	'''
	if clear_current: clear_output(DG)
	for n in DG.nodes(): 
		if len(DG.successors(n)) == 0: #this is an output node
		    computeNodeOutput(DG,n)
		    
def error_back_prop(DG, clear_current = True):
	'''Computes the error term for each node in a feed-forward network, using back
	   propagation, associated with some input vector x and the associated target value t. 
	   It is assumed that the output values for each node, using the input x, have already
	   been computed and are stored in an attribute labeled 'o' for each node.
	   INPUT:
	   	DG - a directed graph
	   	  The nodes of DG must contain the following attributes:
	   	  	daf : the derivative of the node's activation function, which is itself a function
	   	  	      of a single input variable, namely a_j=sum_i[w_ji z_i] where i indexes over 
	   	  	      nodes that send input to node j. For output nodes:
	   	  	   		1. daf should account for the target value t.
	   	  	   		2. will typically have the canonical link function as an activation function so that
	   	  	   		   daf will by a_j - t_j. TyNone, it is assumed (see Bishop page 243)
	   	  	o : the output attribute of the node, should be set based on applying forward_prop for
	   	  	    the input x associated with t. 
	   	  The edges should be weighted edges, with the following attributes:
	         w  : weight
	      clear_current - if True, the algorithm will first traverse DG to clear each node's 'e'
	                    attribute. This is useful for algorithms requiring multiple propagations through 
	                    DG
	  OUTPUT:
	    Every node in DG will have the attribute 'e' added representing the error term at that node
	    computed as daf(a_j)*sum_k(w_kj error_k) where k indexes over nodes that receive input from
	    node j. For output nodes this sum is taken as 1, so that the error term is just daf(a_j)
	'''
	if clear_current: clear_errors(DG)
	for n in DG.nodes():
		if len(DG.predecessors(n)) == 0: #this is an input or bias node
			computeNodeError(DG,n)
			
def computeNodeError(DG, node):
	if DG.node[node]['e']: #this node's error has already been computed
		return DG.node[node]['e']
	else: #need to compute error
		sucs = DG.successors(node)
		preds = DG.predecessors(node)
		ua = reduce(lambda wsum, pred: wsum + DG.edge[pred][node]['weight']*DG.node[pred]['o'], 
		 				preds,
		 				0)
		if len(sucs) > 0: #this is not an output node
			sk = reduce(lambda wsum, suc: wsum + DG.edge[node][suc]['weight']*computeNodeError(DG,suc),
			             sucs,
			             0)
		else: #this is an output node
		    sk = 1.0
		e = DG.node[node]['daf'](ua) * sk
		DG.node[node]['e'] = e
		return e
	
	
def clear_errors(DG):
	for key in DG.nodes():
		DG.node[key]['e'] = None
	
def clear_output(DG):
	for key in DG.nodes():
		DG.node[key]['o'] = None	
	
def computeNodeOutput(DG, node):
	if DG.node[node]['o']: #this node's output has already been computed
		return DG.node[node]['o']
	else: #need to compute output
		#compute unit activation
		preds = DG.predecessors(node)
		if len(preds) > 0:
			ua = reduce(lambda wsum, pred: wsum + DG.edge[pred][node]['weight']*computeNodeOutput(DG,pred), 
		 				preds,
		 				0)
			o = DG.node[node]['af'](ua)
		else: #this node is a bias term and has no input
			o = DG.node[node]['af']
		DG.node[node]['o'] = o
		return o
		
