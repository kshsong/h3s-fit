# pipnn_interface.py
import numpy as np
from PIPNN_PES import pes_init, evvdvdx

_INITIALIZED = False

class PIPNNWrapper:
    def __init__(self):
        pass

    def _check_init(self):
        global _INITIALIZED
        if not _INITIALIZED:
            pes_init()
            _INITIALIZED = True

    def calculate(self, positions: np.ndarray, calc_forces: bool, hx: float = 1.0e-4) -> tuple:
        """
        计算能量和力（数值梯度）
        :param positions: (natoms,3)数组，原子顺序必须严格符合PIP-NN要求
        :param calc_forces: 是否计算受力
        :param hx: 有限差分步长 (默认 1e-4 Å)
        :return: (energy, forces) forces为None如果不计算
        """
        self._check_init()

        natom = positions.shape[0]
        ndim = 3 * natom

        # 先计算未扰动的能量
        xcart = np.asfortranarray(positions.T, dtype=np.float64)
        v = evvdvdx(xcart)  # 不计算解析梯度
        energy = float(v)

        if not calc_forces:
            return energy, None

        # 数值计算梯度（中心差分）
        grad = np.zeros(ndim, dtype=np.float64)
        for a in range(ndim):
            atom_idx = a // 3
            comp_idx = a % 3

            # +hx
            pos_plus = positions.copy()
            pos_plus[atom_idx, comp_idx] += hx
            xcart_plus = np.asfortranarray(pos_plus.T, dtype=np.float64)
            v_plus = evvdvdx(xcart_plus)

            # -hx
            pos_minus = positions.copy()
            pos_minus[atom_idx, comp_idx] -= hx
            xcart_minus = np.asfortranarray(pos_minus.T, dtype=np.float64)
            v_minus = evvdvdx(xcart_minus)

            # 中心差分：dE/dx_a ≈ (E(x+hx) - E(x-hx)) / (2*hx)
            grad[a] = (v_plus - v_minus) / (2.0 * hx)

        # 受力 = -梯度，reshape为(natom, 3)
        forces = -grad.reshape(natom, 3)

        return energy, forces

    def calculate_hessian(self, positions: np.ndarray, hx: float = 1.0e-4) -> tuple:
        """
        计算能量、力和Hessian矩阵（基于数值梯度）
        :param positions: (natoms,3)数组，原子顺序必须严格符合PIP-NN要求
        :param hx: 有限差分步长 (默认 1e-4 Å)
        :return: (energy, forces, hessian)
                 energy: 标量
                 forces: (natoms,3)数组
                 hessian: (3*natoms, 3*natoms) 数组
        """
        self._check_init()

        natom = positions.shape[0]
        ndim = 3 * natom

        # 获取未扰动的能量和数值受力
        energy, forces = self.calculate(positions, calc_forces=True, hx=hx)

        # 数值计算Hessian：对每个坐标扰动，重新计算受力，然后差分
        hess = np.zeros((ndim, ndim), dtype=np.float64)

        for a in range(ndim):
            atom_a = a // 3
            comp_a = a % 3

            # +hx
            pos_plus = positions.copy()
            pos_plus[atom_a, comp_a] += hx
            _, forces_plus = self.calculate(pos_plus, calc_forces=True, hx=hx)
            grad_plus = forces_plus.flatten()  # 注意：forces是(natom,3)，flatten后是(ndim,)

            # -hx
            pos_minus = positions.copy()
            pos_minus[atom_a, comp_a] -= hx
            _, forces_minus = self.calculate(pos_minus, calc_forces=True, hx=hx)
            grad_minus = forces_minus.flatten()

            # Hessian[a, :] = d(force)/dx_a ≈ (force(x+hx) - force(x-hx)) / (2*hx)
            # 注意：force = -grad(E)，所以 Hessian[i,j] = - d²E/(dx_i dx_j)
            hess[a, :] = (grad_plus - grad_minus) / (2.0 * hx)

        # 对称化（推荐）
        hess = 0.5 * (hess + hess.T)

        return energy, forces, hess
