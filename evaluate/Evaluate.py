#!/usr/bin/env python3
"""
Unified ASE interface using PIP-NN calculator (PIPNN_calculator).

Supports:
 --ene   : single-point energy calculation (default)
 --optg  : geometry optimization (BFGS)
 --optt  : transition state optimization (MinModeAtoms + FIRE, works in ASE 3.22.1)
 --freq  : vibrational frequency calculation

Input xyz format:
 line 1  : number of atoms
 line 2  : ab-initio energy in eV
 line 3+ : symbol  x  y  z
 Multiple frames may be concatenated.
"""

import argparse, math, sys, os
from os.path import splitext
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.io import write
from ase.optimize import BFGS,LBFGS,FIRE
from ase.mep import MinModeAtoms
from ase.vibrations import Vibrations
try:
    from ase.vibrations import VibrationsData
except ImportError:
    from ase.vibrations.data import VibrationsData
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sella import Sella

# Import PIPNN Calculator
try:
    from pipnn_calculator import PIPNNCalculator
except Exception as e:
    PIPNNCalculator = None
    print("Warning: Could not import PIPNNCalculator:", e, file=sys.stderr)

EV_TO_KCAL = 23.060548  # 1 eV = 23.060548 kcal/mol

# --------- Utility functions ---------
def read_xyz_with_energies(filename):
    frames = []
    with open(filename) as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 0
    while i < len(lines):
        nat = int(lines[i]); comment = lines[i+1]
        try:
            energy = float(comment.split()[0])
        except:
            energy = 0.0
        coords, syms = [], []
        for j in range(nat):
            parts = lines[i+2+j].split()
            syms.append(parts[0])
            coords.append(list(map(float, parts[1:4])))
        atoms = Atoms(symbols=syms, positions=coords)
        frames.append((atoms, energy))
        i += nat + 2
    return frames

def create_calculator(atoms):
    if PIPNNCalculator is None:
        raise RuntimeError("PIPNNCalculator not available")
    try:
        return PIPNNCalculator()
    except Exception as e:
        raise RuntimeError("Failed to init PIPNNCalculator") from e

def calculate_rmsd_kabsch(posA, posB):
    A, B = np.array(posA), np.array(posB)
    A -= A.mean(axis=0); B -= B.mean(axis=0)
    C = A.T @ B
    V, _, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    U = V @ np.diag([1,1,d]) @ Wt
    A_rot = A @ U
    return np.sqrt(((A_rot - B)**2).sum()/A.shape[0])

def get_bond_length(p, i,j): return np.linalg.norm(p[i]-p[j])
def get_angle_deg(p,i,j,k):
    v1, v2 = p[i]-p[j], p[k]-p[j]
    cosang = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    return math.degrees(math.acos(np.clip(cosang,-1,1)))
def get_dihedral_deg(p,i,j,k,l):
    b0,b1,b2 = p[j]-p[i], p[k]-p[j], p[l]-p[k]
    n0,n1 = np.cross(b0,b1), np.cross(b1,b2)
    n0/=np.linalg.norm(n0); n1/=np.linalg.norm(n1)
    m1 = np.cross(n0,b1/np.linalg.norm(b1))
    return math.degrees(math.atan2(np.dot(m1,n1), np.dot(n0,n1)))

# --------- Mode implementations ---------
def single_point_calculation(args):
    frames = read_xyz_with_energies(args.input)
    energies_true, energies_pred, params = [], [], []
    param_type, idx = None, None
    if args.bond: param_type, idx = 'bond',[i-1 for i in args.bond]
    if args.angle: param_type, idx = 'angle',[i-1 for i in args.angle]
    if args.dihedral: param_type, idx = 'dihedral',[i-1 for i in args.dihedral]

    #Create the calculator ONCE before the loop
    if not frames: return
    #calc = create_calculator(frames[0][0]); 

    for atoms, e in frames:
        atoms.calc = create_calculator(atoms)
        e_pred = atoms.get_potential_energy()
        energies_true.append(e); energies_pred.append(e_pred)
        if param_type:
            p = atoms.get_positions()
            if param_type=='bond': params.append(get_bond_length(p,*idx))
            if param_type=='angle': params.append(get_angle_deg(p,*idx))
            if param_type=='dihedral': params.append(get_dihedral_deg(p,*idx))

    if len(frames)==1 and not param_type:
        print(f"Ab-initio: {energies_true[0]:.6f} eV ({energies_true[0]*EV_TO_KCAL:.6f} kcal/mol)")
        print(f"PIPNN    : {energies_pred[0]:.6f} eV ({energies_pred[0]*EV_TO_KCAL:.6f} kcal/mol)")
        return
    # ========== 新增：将多帧能量数据输出到文件 ==========
    if len(frames) > 1:
        output_filename = splitext(args.input)[0] + "-compare.txt"
        with open(output_filename, 'w') as f:
            # 写入表头
            f.write(f"{'Frame':>5} {'Ab-initio (eV)':>15} {'PIPNN (eV)':>15} {'Error (eV)':>15}\n")
            f.write("-" * 55 + "\n")
            # 写入每一帧的数据
            for i, (e_true, e_pred) in enumerate(zip(energies_true, energies_pred), start=1):
                error = e_pred - e_true
                f.write(f"{i:>5} {e_true:>15.6f} {e_pred:>15.6f} {error:>15.6f}\n")
        print(f"Energy comparison saved to: {output_filename}")
    # ====================================================
    if param_type:
        y_true, y_pred = np.array(energies_true)*EV_TO_KCAL, np.array(energies_pred)*EV_TO_KCAL
        plt.plot(params,y_pred,'k-',label='PIP-NN')
        plt.scatter(params,y_true,c='r',label='ab-initio')
        plt.xlabel(param_type)
        plt.ylabel("Energy (kcal/mol)")
        plt.legend()
        plt.tight_layout()
        out = splitext(args.input)[0]+f"-{param_type}-energy.png"
        plt.savefig(out,dpi=300)
        print("Saved",out)
        plt.show()
        return
    rmse=np.sqrt(mean_squared_error(energies_true,energies_pred))
    mae=mean_absolute_error(energies_true,energies_pred)
    plt.scatter(energies_true, energies_pred,c='k',label='PIP-NN')
    minv,maxv=min(energies_true+energies_pred),max(energies_true+energies_pred)
    plt.plot([minv,maxv],[minv,maxv],'--',c='gray')
    plt.xlabel("ab-initio (eV)"); plt.ylabel("PIPNN (eV)")
    plt.text(0.98,0.02,f"RMSE={rmse:.4f}\nMAE={mae:.4f}",transform=plt.gca().transAxes,ha='right',va='bottom')
    out = splitext(args.input)[0]+"-corr.png"
    plt.savefig(out,dpi=300)
    print("Saved",out)
    plt.show()

def geometry_optimization(args):
    atoms,_ = read_xyz_with_energies(args.input)[0]
    init=atoms.get_positions().copy()
    atoms.calc = create_calculator(atoms)
    #You can use Sella here also.
    # opttraj=splitext(args.input)[0]+'-optg.traj'
    # dyn=Sella(atoms, order=0,trajectory='optg.traj',logfile=None)
    # dyn.run(fmax=args.fmax, steps=100)
    dyn = LBFGS(atoms=atoms)
    dyn.run(fmax=args.fmax, steps=100)

    out=splitext(args.input)[0]+"-optg.xyz"
    write(out,atoms)
    print("Saved",out)
    print("RMSD:",calculate_rmsd_kabsch(init,atoms.get_positions()))

    return atoms

def transition_state_optimization(args):
    atoms,_ = read_xyz_with_energies(args.input)[0]
    init=atoms.get_positions().copy()
    atoms.calc = create_calculator(atoms)
    opttraj=splitext(args.input)[0]+'-optt.traj'
    out=splitext(args.input)[0]+'-optg.xyz'
    dyn=Sella(atoms,order=1,trajectory=opttraj,logfile=None)
    dyn.run(fmax=args.fmax, steps=100)
    write(out,atoms)
    print("Saved",out)
    print("RMSD:",calculate_rmsd_kabsch(init,atoms.get_positions()))
    
    return atoms

def frequency_calculation(args):
    atoms, _ = read_xyz_with_energies(args.input)[0]
    atoms.calc = create_calculator(atoms)

    if args.type == 'ts':
        atoms = transition_state_optimization(args)
    else:
        atoms = geometry_optimization(args)

    if atoms.calc is None:
        atoms.calc = create_calculator(atoms)

    vibname = splitext(args.input)[0] + "-vib"
    vib = Vibrations(atoms, delta=0.01, nfree=4, name=vibname)
    vib.run()
    vib.summary()
    freqs = vib.get_frequencies()

    n_modes_skip = 6 if len(atoms) > 2 else 5

    if args.type == 'ts':
        # For transition states, identify significant imaginary frequencies (>100 cm⁻¹)
        imaginary_freqs = freqs[np.abs(freqs.imag) > 100]  # Filter for significant imaginary frequencies
        # Extract real frequencies, skipping translational/rotational modes
        real_freqs = freqs[n_modes_skip:]
        real_freqs = real_freqs[np.abs(real_freqs.imag) < 1e-6]  # Keep only real frequencies
        real_freqs = real_freqs.real  # Convert to real numbers for cleaner output
        
        if len(imaginary_freqs) > 0:
            print("Transition state imaginary frequencies (cm⁻¹):", imaginary_freqs)
            freqs = np.concatenate([real_freqs, imaginary_freqs])    
        else:
            print("Warning: No significant imaginary frequencies (>100 cm⁻¹) found for transition state.")
            freqs = real_freqs
        print("Transition state real frequencies (cm⁻¹):", real_freqs)
    else:
        # For non-transition states, expect no significant imaginary frequencies
        real_freqs = freqs[n_modes_skip:]
        real_freqs = real_freqs[np.abs(real_freqs.imag) < 1e-6]  # Keep only real frequencies
        real_freqs = real_freqs.real  # Convert to real numbers
        imaginary_freqs = freqs[np.abs(freqs.imag) > 100]  # Check for significant imaginary frequencies
        if len(imaginary_freqs) > 0:
            print("Warning: Unexpected significant imaginary frequencies (>100 cm⁻¹) found for non-transition state (cm⁻¹):", imaginary_freqs)
        else:
            print("No transition state. No significant imaginary frequencies (>100 cm⁻¹) expected.")
        print("Real frequencies (cm⁻¹):", real_freqs)
        freqs = real_freqs

    if args.orifreq and os.path.exists(args.orifreq):
        ref = [float(l.split()[0]) for l in open(args.orifreq) if l.strip()]
        n = min(len(ref), len(freqs))
        ref, freqs = np.array(ref[:n]), freqs[:n]

        rmse = np.sqrt(mean_squared_error(ref, freqs))
        mae = mean_absolute_error(ref, freqs)

        plt.scatter(ref, freqs, c='k')
        plt.plot([min(ref), max(ref)], [min(ref), max(ref)], '--', c='gray')
        plt.xlabel("Reference frequencies (cm$^{-1}$)")
        plt.ylabel("ASE frequencies (cm$^{-1}$)")
        plt.title("Frequency correlation")
        plt.text(0.98, 0.02, f"RMSE={rmse:.2f}\nMAE={mae:.2f}",
                 transform=plt.gca().transAxes, ha='right', va='bottom')

        out = splitext(args.input)[0] + "-freq-corr.png"
        plt.savefig(out, dpi=300)
        print("Saved", out)
        plt.show()

    vib.clean()
    os.removedirs(vibname)

# --------- Main ---------
def main():
    p=argparse.ArgumentParser()
    p.add_argument("-i","--input",required=True)
    m=p.add_mutually_exclusive_group()
    m.add_argument("--ene",action="store_true"); m.add_argument("--optg",action="store_true")
    m.add_argument("--optt",action="store_true"); m.add_argument("--optfreq",action="store_true")
    p.add_argument("--type", choices=['min', 'ts'], default='min', help="Specify if structure is minima ('min') or transition state ('ts') for optimization")
    p.add_argument("--charge",type=float,default=0.0); p.add_argument("--fmax",type=float,default=None)
    p.add_argument("--bond",nargs=2,type=int); p.add_argument("--angle",nargs=3,type=int); p.add_argument("--dihedral",nargs=4,type=int)
    p.add_argument("--orifreq")
    a=p.parse_args()
    if not any([a.ene,a.optg,a.optt,a.optfreq]): a.ene=True
    if a.fmax is None: a.fmax=5e-4 if a.optg else (0.05 if a.optt else 5e-4)
    if a.ene: single_point_calculation(a)
    if a.optg: geometry_optimization(a)
    if a.optt: transition_state_optimization(a)
    if a.optfreq: frequency_calculation(a)

if __name__=="__main__": main()

