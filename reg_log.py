import nltk
import numpy as np
import spacy
import random
import math

def get_data(file):
	f = open(file)
	mat = []
	for line in f:
		sentence = nltk.word_tokenize(line)
		sentence = [word.lower() for word in sentence]
		mat.append(sentence)
	f.close()
	return mat

def shuffle(data, porcentage = 0.7):
	random.shuffle(data,random.random)
	lim = int(len(data)*porcentage)
	train = data[:lim]
	test = data[lim:]
	return [train,test]

def get_y(data):
	y = []
	for row in data:
		y.append(row[-1])
	return np.array(y)

def get_x(data):
	x = []
	for row in data:
		x.append(row[:-2])
	return x

def get_num_y(y):
	num_y = []
	for word in y:
		if word == "spam":
			num_y.append(1)
		else:
			num_y.append(0)
	return np.array(num_y)

def lem_x(x):
	lem_x = []
	lem_row = []
	nlp = spacy.load("en_core_web_sm")

	for row in x:
		lem_row = []
		for token in row:
			doc = nlp(token)
			lem = doc[0].lemma_
			lem_row.append(lem)
		lem_x.append(lem_row)
	return lem_x

def get_voc(lem_x):
	voc = []
	for row in lem_x:
		for word in row:
			if voc.count(word) == 0:
				voc.append(word)
	return sorted(voc)

def get_num_x(lem_x):
	voc = get_voc(lem_x)
	vs = len(voc)
	x_num = []
	for row in lem_x:
		vec_freq = [1] #insert x0 = 1
		for word in voc:
			f = row.count(word)
			vec_freq.append(f)
		vec_freq = np.array(vec_freq)
		vec_freq = vec_freq / vec_freq.sum()
		x_num.append(vec_freq)
	return x_num

def hypothesis(x,theta):
	#print(type(x),type(theta))
	x = np.array(x)
	theta = np.array(theta)
	z = x.dot(theta)
	return 1/(1 + (math.e ** (-z)))

def cost(x,theta,y):
	if y == 1:
		return -math.log(hypothesis(x,theta))
	else:
		return -math.log(1 - hypothesis(x,theta))

def jcost(x,theta,y,m):
	su = 0
	for i in range(m):
		su = su + cost(x[i],theta,y[i])
	return su/m

def make_vec(num,fit = 0):
	vec = []
	for i in range(num):
		vec.append(fit)
	return vec

def suma(x,theta,y,m,j):
	su = 0
	for i in range(m):
		su = su + (hypothesis(x[i],theta) - y[i]) * x[i][j]
	return su

def min_cost(x,y,alpha = 0.1,iterations = 1000, show = 50):
	n = len(x[0])
	m = len(y)
	theta = make_vec(n)
	aux = make_vec(n)
	costs = []

	for ite in range(iterations):
		#print("iteration",ite)
		if ((ite + 1)%iterations == 0):
			c = jcost(x,theta,y,m)
			cost.append(c)
			print("Cost at iteration", (ite + 1), "= ", c)
			
		for j in range(n):
			#print("j",j)
			aux[j] = aux[j] - alpha * suma(x,theta,y,m,j)
			#print(aux[j])
		theta = aux

	return theta, cost

def make_predictions(theta,x,y):
	f = open("predictions.txt","w")
	for i in range(len(y)):
		f.write("Prediction:",hypothesis(x[i],theta), "Correct:", y[i])
	f.close()

if __name__ == '__main__':
	'''x = [1,0,0,0.5,0.33]
	t = [1,1,1,0,0]
	print(hypothesis(x,t))
	print(cost(x,t,0))'''
	
	data = get_data("data.txt")
	train, test = shuffle(data)
	#print(data[0])
	y = get_y(data)
	#print(y[:10])
	x = get_x(data)
	#print(x[0])
	#print(len(y))
	#print(len(x))
	ny = get_num_y(y)
	#print(ny[:10])
	lx = lem_x(x)
	#print(lx[0])
	nx = get_num_x(lx)
	#print(len(x),len(lx),len(nx),nx[0])
	theta, costs = min_cost(nx,ny)
	make_predictions(theta, x, y)
	