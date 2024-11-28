graph ={
 '5':["3","7"] ,
 '3':["2","4"] ,
 '7':["8"] ,
 '2':[ ] ,
 '4': ["8"],
 '8':[ ]
}

def dls(node, graph ,goal , depth ,visited):
	print(node, end=" ")
	if node==goal:
		return True
	if depth<=0:
	  return False
	visited.append(node)
	for neighbour in graph[node]:
	  		if neighbour not in visited:
	  			if dls(neighbour,graph,goal,depth-1,visited):
	  				return True
	return False
   
def ids(graph, start ,goal ,max_depth):
	  for depth in range(max_depth):
	  	visited=[]
	  	print("\ndepth level:",depth)
	  	if dls(start,graph,goal,depth,visited):
	  		print("\n goal found!")
	  		return
	  print("\n Goal not found in depth limit")
	  
print("Following is the iterative deapining learning")
ids(graph, '5' , '8' , 5)	  
