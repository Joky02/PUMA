import os
import json
from typing import Callable, Optional, Any, List
import gurobipy as gp


# Define a unified model interface
class BaseModelWrapper:
	def __init__(self, ins_path: str, log_file: str, **params):
		self.ins_path = ins_path
		self.log_file = log_file
		self.params = params

	def add_var(self, *args, **kwargs) -> Any:
		"""Add a variable. Common kwargs: name, lb, ub, vtype, obj"""
		raise NotImplementedError

	def add_constr(self, *args, **kwargs) -> Any:
		"""Add a constraint. Common args: expr, name"""
		raise NotImplementedError

	def set_params(self, **params):
		"""Set solver parameters"""
		raise NotImplementedError

	def get_vars(self) -> List[Any]:
		"""Return list of decision variables"""
		raise NotImplementedError

	def set_start(self, var: Any, value: float):
		"""Set MIP start for a variable"""
		raise NotImplementedError

	def optimize(self) -> None:
		"""Run optimization"""
		raise NotImplementedError

	def extract_results(self) -> dict:
		"""Extract results into a dict"""
		raise NotImplementedError


class GurobiWrapper(BaseModelWrapper):
	def __init__(self, ins_path: str, log_file: str, **params):
		super().__init__(ins_path, log_file, **params)
		self.model = gp.read(ins_path)
		os.makedirs(os.path.dirname(log_file), exist_ok=True)
		self.model.setParam('LogFile', log_file)

	def add_var(self, name=None, lb=0.0, ub=1.0, vtype=gp.GRB.CONTINUOUS, obj=0.0) -> Any:
		return self.model.addVar(name=name, lb=lb, ub=ub, vtype=vtype, obj=obj)

	def add_constr(self, expr, name=None) -> Any:
		return self.model.addConstr(expr, name=name)

	def set_params(self, **params):
		for k, v in params.items():
			self.model.setParam(k, v)

	def get_vars(self) -> List[Any]:
		return self.model.getVars()

	def set_start(self, var: Any, value: float):
		var.Start = value

	def optimize(self):
		self.model.optimize()

	def extract_results(self) -> dict:
		return {
			"obj": self.model.ObjVal,
			"time": self.model.Runtime,
			"nnodes": self.model.NodeCount,
			"gap": self.model.MIPGap,
			"stat": self.model.status
		}


class SCIPWrapper(BaseModelWrapper):
	def __init__(self, ins_path: str, log_file: str, **params):
		from pyscipopt import Model
		super().__init__(ins_path, log_file, **params)
		self.model = Model()
		self.model.readProblem(ins_path)
		os.makedirs(os.path.dirname(log_file), exist_ok=True)

		lf = open(log_file, 'w')

		self.model.setLogfile(log_file)
		self._log_fp = lf
		self._scip_sol = None  # For SCIP

	def add_var(self, name=None, lb=0.0, ub=1.0, vtype='C', obj=0.0) -> Any:
		return self.model.addVar(name, vtype=vtype, lb=lb, ub=ub, obj=obj)

	def add_constr(self, expr, name=None) -> Any:
		return self.model.addCons(expr, name)

	def set_params(self, **params):
		for k, v in params.items():
			self.model.setParam(k, v)

	def get_vars(self) -> List[Any]:
		return self.model.getVars()

	def set_start(self, var: Any, value: float):
		# pyscipopt: use setStartVal
		if self._scip_sol is None:
			self._scip_sol = self.model.createSol()
		self.model.setSolVal(self._scip_sol, var, value)

	def optimize(self):
		if self._scip_sol is not None:
			self.model.addSol(self._scip_sol)
		self.model.optimize()
		self._log_fp.close()

	def extract_results(self) -> dict:
		return {
			"obj": self.model.getObjVal(),
			"time": self.model.getSolvingTime(),
			"nnodes": self.model.getNNodes(),
			"gap": self.model.getGap(),
			"stat": self.model.getStatus()
		}
