# pipnn_calculator.py
from ase.calculators.calculator import Calculator, all_changes
import numpy as np
from pipnn_interface import PIPNNWrapper

class PIPNNCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'hessian']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wrapper = PIPNNWrapper()

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        # 关键约束：必须保证原子顺序与PIP-NN定义一致！
        if atoms is not None:
            positions = atoms.get_positions()
            self.atoms = atoms.copy()
        else:
            positions = self.atoms.get_positions()

        
        if 'hessian' in properties:
            energy, forces, hessian = self.wrapper.calculate_hessian(positions)
            self.results={
                'energy': energy,
                'forces': forces,
                'hessian': hessian
            }
        elif 'energy' in properties and 'forces' in properties:
            energy, forces = self.wrapper.calculate(positions, calc_forces=True)
            self.results["energy"] = energy
            self.results["forces"] = forces
        elif 'forces' not in properties:
            energy, forces = self.wrapper.calculate(positions, calc_forces=False)


    def get_property(self, name, atoms=None, allow_calculation=True):
        """获取属性，兼容ASE接口"""
        if atoms is not None:
            self.atoms = atoms

        self.results = {}
        if name == "hessian":
            self.calculate(atoms=self.atoms, properties=["energy", "forces", "hessian"])
        else:
            self.calculate(atoms=self.atoms, properties=["energy", "forces"])
        
        return self.results.get(name)

    # 保持向后兼容的方法
    def get_potential_energy(self, atoms=None, force_consistent=False):
        return self.get_property('energy', atoms)
    
    def get_forces(self, atoms=None):
        return self.get_property('forces', atoms)
    
    def get_hessian(self, atoms=None):
        return self.get_property('hessian', atoms)
    # 明确实现 hessian 属性计算方法
    def _get_hessian(self, atoms=None):
        """ASE 内部调用的 hessian 计算方法"""
        if atoms is not None:
            self.atoms = atoms
        if 'hessian' not in self.results:
            self.calculate(atoms=self.atoms, properties=['hessian'])
        return self.results['hessian']

