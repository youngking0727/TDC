import pandas as pd
import numpy as np
import os, sys, json 
import warnings
warnings.filterwarnings("ignore")

from ..utils import *
from . import bi_pred_dataset, multi_pred_dataset
from ..metadata import dataset_names

class DTI(bi_pred_dataset.DataLoader):
	def __init__(self, name, path = './data', label_name = None, print_stats = False):
		super().__init__(name, path, label_name, print_stats, dataset_names = dataset_names["DTI"])
		self.entity1_name = 'Drug'
		self.entity2_name = 'Target'
		
		if print_stats:
			self.print_stats()

		print('Done!', flush = True, file = sys.stderr)

class DDI(bi_pred_dataset.DataLoader):
	def __init__(self, name, path = './data', label_name = None, print_stats = False):
		super().__init__(name, path, label_name, print_stats, dataset_names = dataset_names["DDI"])
		self.entity1_name = 'Drug1'
		self.entity2_name = 'Drug2'
		
		if print_stats:
			self.print_stats()

		print('Done!', flush = True, file = sys.stderr)

	def print_stats(self):
		print_sys('--- Dataset Statistics ---')
		print('There are ' + str(len(np.unique(self.entity1.tolist() + self.entity2.tolist()))) + ' unique drugs.', flush = True, file = sys.stderr)
		print('There are ' + str(len(self.y)) + ' drug-drug pairs.', flush = True, file = sys.stderr)
		print_sys('--------------------------')

class PPI(bi_pred_dataset.DataLoader):
	def __init__(self, name, path = './data', label_name = None, print_stats = False):
		super().__init__(name, path, label_name, print_stats, dataset_names = dataset_names["PPI"])
		self.entity1_name = 'Protein1'
		self.entity2_name = 'Protein2'
		
		if print_stats:
			self.print_stats()

		print('Done!', flush = True, file = sys.stderr)

	def print_stats(self):
		print_sys('--- Dataset Statistics ---')
		print('There are ' + str(len(np.unique(self.entity1.tolist() + self.entity2.tolist()))) + ' unique proteins.', flush = True, file = sys.stderr)
		print('There are ' + str(len(self.y)) + ' protein-protein pairs.', flush = True, file = sys.stderr)
		print_sys('--------------------------')

class PeptideMHC(bi_pred_dataset.DataLoader):
	def __init__(self, name, path = './data', label_name = None, print_stats = False):
		super().__init__(name, path, label_name, print_stats, dataset_names = dataset_names["PeptideMHC"])
		self.entity1_name = 'Peptide'
		self.entity2_name = 'MHC'
		
		if print_stats:
			self.print_stats()

		print('Done!', flush = True, file = sys.stderr)

class MTI(bi_pred_dataset.DataLoader):
	def __init__(self, name, path = './data', label_name = None, print_stats = False):
		super().__init__(name, path, label_name, print_stats, dataset_names = dataset_names["MTI"])
		self.entity1_name = 'miRNA'
		self.entity2_name = 'Target'
		
		if print_stats:
			self.print_stats()

		print('Done!', flush = True, file = sys.stderr)

class GDA(bi_pred_dataset.DataLoader):
	def __init__(self, name, path = './data', label_name = None, print_stats = False):
		super().__init__(name, path, label_name, print_stats, dataset_names = dataset_names["GDA"])
		self.entity1_name = 'Gene'
		self.entity2_name = 'Disease'
		
		if print_stats:
			self.print_stats()

		print('Done!', flush = True, file = sys.stderr)

class AntibodyAff(bi_pred_dataset.DataLoader):
	def __init__(self, name, path = './data', label_name = None, print_stats = False):
		super().__init__(name, path, label_name, print_stats, dataset_names = dataset_names["AntibodyAff"])
		self.entity1_name = 'Antibody'
		self.entity2_name = 'Antigen'
		
		if print_stats:
			self.print_stats()

		print('Done!', flush = True, file = sys.stderr)

class DrugRes(bi_pred_dataset.DataLoader):
	def __init__(self, name, path = './data', label_name = None, print_stats = False):
		super().__init__(name, path, label_name, print_stats, dataset_names = dataset_names["DrugRes"])
		self.entity1_name = 'Drug'
		self.entity2_name = 'Cell Line'
		
		if print_stats:
			self.print_stats()

		print('Done!', flush = True, file = sys.stderr)

class DrugSyn(multi_pred_dataset.DataLoader):
	def __init__(self, name, path = './data', print_stats = False):
		super().__init__(name, path, print_stats, dataset_names = dataset_names["DrugSyn"])

		if print_stats:
			self.print_stats()

		print('Done!', flush = True, file = sys.stderr)


class Catalyst(bi_pred_dataset.DataLoader):
	def __init__(self, name, path = './data', label_name = None, print_stats = False):
		super().__init__(name, path, label_name, print_stats, dataset_names = dataset_names["Catalyst"])
		self.entity1_name = 'Reactant'
		self.entity2_name = 'Product'
		if print_stats:
			self.print_stats()

		print('Done!', flush = True, file = sys.stderr)
