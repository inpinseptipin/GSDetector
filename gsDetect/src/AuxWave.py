import wave
import os
import re


class WaveLoader:
	def __init__(self,path):
		fileList = os.listdir(path)
		self.fileList = []
		for f in fileList:
			if re.search('.wav',f):
				self.fileList +=[f]				

	def getList(self):
		return self.fileList


Loader = WaveLoader("")
files = Loader.getList()
print(files)