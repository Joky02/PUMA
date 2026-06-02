import json

import torch
import os
import random
import numpy as np
import pyscipopt as scp
from torch_geometric.data import Batch, Data
from typing import Callable, Optional, Any
from src.model import BipartiteNodeData

from src.solver_wrapper import BaseModelWrapper, GurobiWrapper, SCIPWrapper


def set_seed(seed):
	"""
	Set the random seed.
	"""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)


def set_cpu_num(cpu_num):
	"""
	Set the number of used cpu kernals.
	"""
	os.environ['OMP_NUM_THREADS'] = str(cpu_num)
	os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
	os.environ['MKL_NUM_THREADS'] = str(cpu_num)
	os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
	os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
	torch.set_num_threads(cpu_num)


def get_A_b_c(ins_name, sparse: bool = False):
	m = scp.Model()
	m.hideOutput(True)
	m.readProblem(ins_name)

	ncons = m.getNConss()
	nvars = m.getNVars()

	vars = m.getVars()
	vars.sort(key=lambda v: v.name)

	v_map = {}

	for indx, v in enumerate(vars):
		v_map[v.name] = indx

	indices_spr = [[], []]
	values_spr = []

	cons = []
	for cind, c in enumerate(m.getConss()):
		coeff = m.getValsLinear(c)
		if len(coeff) != 0:
			cons.append(c)

	cons_map = [[x, len(m.getValsLinear(x))] for x in cons]

	cons_map = sorted(cons_map, key=lambda x: [x[1], str(x[0])])
	cons = [x[0] for x in cons_map]
	cons_count = 0
	c_nodes = []
	for cind, c in enumerate(cons):
		coeff = m.getValsLinear(c)
		rhs = m.getRhs(c)
		lhs = m.getLhs(c)
		sense = 0

		if rhs == lhs:
			sense = 2
			cons_count += 1		
		elif rhs >= 1e+20:
			sense = 1
			rhs = lhs

		for k in coeff:
			v_indx = v_map[k]
			if coeff[k] != 0:
				if sense == 1:
					indices_spr[0].append(cons_count)
					indices_spr[1].append(v_indx)
					values_spr.append(-coeff[k])
				if sense == 0:
					indices_spr[0].append(cons_count)
					indices_spr[1].append(v_indx)
					values_spr.append(coeff[k])
				if sense == 2:
					indices_spr[0].append(cons_count - 1)
					indices_spr[1].append(v_indx)
					values_spr.append(coeff[k])
					indices_spr[0].append(cons_count)
					indices_spr[1].append(v_indx)
					values_spr.append(-coeff[k])
     
		cons_count += 1		

			
	indices_spr = torch.as_tensor(indices_spr, dtype=torch.int64)
	values_spr = torch.as_tensor(values_spr, dtype=torch.float32)

	A = torch.sparse_coo_tensor(indices_spr, values_spr, (cons_count, nvars))

	# build constraint right-hand side b
	b = []
	for c in cons:
		rhs = m.getRhs(c)
		lhs = m.getLhs(c)
		if rhs == lhs:
        # aᵀx = rhs  →  [ Ax ≤ rhs,  -Ax ≤ -rhs ]
			b.append(rhs)
			b.append(-rhs)
		elif rhs >= 1e+20:
			# aᵀx ≥ lhs  →  -Ax ≤ -lhs
			b.append(-lhs)
		else:
			# aᵀx ≤ rhs
			b.append(rhs)
	
	b = torch.tensor(b, dtype=torch.float32).unsqueeze(1)

	# build objective vector c
	c = torch.zeros(len(vars), dtype=torch.float32)
	obj = m.getObjective()
	for e in obj:
		vnm = e.vartuple[0].name
		v = obj[e]
		v_indx = v_map[vnm]
		c[v_indx] = v

	if m.getObjectiveSense() == "maximize":
		c = -c
	c = c

	if sparse:
		return A.coalesce(), b, c
	return A.to_dense(), b, c

def solve_problem(
	test_ins_name: str,
	test_data_dir: str,
	log_dir: str,
	json_dir: str,
	solver: str = 'gurobi',
	time_limit: int = 1000,
	threads: int = 1,
	mip_focus: int = 1,
	model_modifier: Optional[Callable[[BaseModelWrapper], None]] = None
) -> dict:
	"""
	Unified solve function supporting Gurobi and SCIP with a common API.

	- model_modifier: function receiving a BaseModelWrapper to add vars/constrs
	"""
	ins_path = os.path.join(test_data_dir, test_ins_name)
	os.makedirs(log_dir, exist_ok=True)
	os.makedirs(json_dir, exist_ok=True)
	log_file = os.path.join(log_dir, f'{test_ins_name}.log')


	if solver.lower() == 'gurobi':
		wrapper = GurobiWrapper(ins_path, log_file)
		# set solver-specific params
		wrapper.set_params(LogToConsole=1, TimeLimit=time_limit, Threads=threads, MIPFocus=mip_focus)

	elif solver.lower() == 'scip':
		from pyscipopt import SCIP_PARAMSETTING
		wrapper = SCIPWrapper(ins_path, log_file)
		# set basic SCIP params
		wrapper.model.setParam("limits/time", time_limit)
		wrapper.model.setParam("parallel/maxnthreads", threads)
		# map mip_focus to SCIP heuristic setting
		if mip_focus == 1:
			# equivalent to setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)
			wrapper.model.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)
	else:
		raise ValueError(f"Unsupported solver '{solver}'.")

	# apply user modifications
	if model_modifier:
		model_modifier(wrapper)

	# optimize
	wrapper.optimize()

	# extract & save
	result = wrapper.extract_results()
	json_path = os.path.join(json_dir, f'{test_ins_name}.json')
	with open(json_path, 'w') as f:
		json.dump(result, f, indent=2)
	return result




def gumbel_sample(logits: torch.Tensor, N: int, tau: float = 1.0):
	logits = logits.reshape(-1, 1)
	logits = logits.repeat(N, 1, 1)
	logits = torch.cat([torch.zeros_like(logits), logits], dim=-1)
	return torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=True)[:, :, 1]

def custom_collate_fn(data_list):
	"""
	Custom collate function for graph data that carries variable-sized
	tensors (A, b, c) which PyG cannot batch automatically.
	"""
	problem_tensors_list = []
	clean_data_list = []

	# 1. Separate the graph data from the non-batchable tensors.
	for data in data_list:
		# Store A, b, c in a separate list.
		problem_tensors_list.append([data.A, data.b, data.c])

		# Build a clean Data object holding only the graph structure that
		# PyG can safely batch, without mutating the original data object.
		clean_data = BipartiteNodeData(
			constraint_features=data.constraint_features,
			edge_indices=data.edge_index,
			edge_features=data.edge_attr,
			variable_features=data.variable_features
		)
		clean_data.num_nodes = data.num_nodes
		clean_data_list.append(clean_data)

	# 2. Batch only the clean graph data using PyG's standard method.
	batch = Batch.from_data_list(clean_data_list, follow_batch=['variable_features', 'constraint_features'], exclude_keys=['A', 'b', 'c'])

	# 3. Re-attach the separated non-batchable tensors as a list attribute.
	batch.problem_tensors = problem_tensors_list

	return batch
