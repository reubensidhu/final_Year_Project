import pickle

data = pickle.load( open( "save.p", "rb" ) )

print(data[3])