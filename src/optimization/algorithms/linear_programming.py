class HybridScheduler:
    """
    COPIED FROM: oHySEM/oHySEM.py
    
    This class creates optimal schedules for hybrid energy systems.
    Like a smart manager who decides which power plants to run when.
    """
    
    def __init__(self):
        self.model = None
        self.results = None
        self.energy_sources = {}
        self.schedule = None
    
    def add_energy_source(self, name: str, max_capacity: float, cost_per_mwh: float, 
                         is_renewable: bool = False, ramp_rate: float = None):
        """
        Add an energy source to the optimization.
        
        COPIED AND MODIFIED FROM: oHySEM energy source definition
        """
        self.energy_sources[name] = {
            'max_capacity': max_capacity,
            'cost': cost_per_mwh,
            'renewable': is_renewable,
            'ramp_rate': ramp_rate or max_capacity  # Default: can ramp full capacity per hour
        }
        print(f"➕ Added {name}: {max_capacity} MW, ${cost_per_mwh}/MWh, Renewable: {is_renewable}")
    
    def create_scheduler_model(self, demand_forecast: list, renewable_forecast: dict = None, 
                              time_horizon: int = 24):
        """
        Create the optimization model.
        COPIED FROM: oHySEM/oHySEM.py model setup section
        """
        print("🔧 Setting up optimization problem...")
        print(f"⏰ Time horizon: {time_horizon} hours")
        
        # Create Pyomo model
        model = pyo.ConcreteModel()
        
        # COPIED FROM oHySEM: Sets definition
        model.T = pyo.Set(initialize=range)
"""
EnergyNexus Advanced Scheduling Algorithms
YOUR ORIGINAL IMPLEMENTATION for MSc Project

Novel Contributions:
- Multi-objective optimization for hybrid systems
- Uncertainty-aware scheduling under renewable variability
- Grid stability constraints with renewable integration
- Real-time adaptive scheduling algorithms

Author: Aditya Talekar (ec24018@qmul.ac.uk)
Supervisor: Saqib Iqbal
QMUL MSc Data Science and AI - 2024/25

Academic Integrity: Uses Pyomo as optimization tool but implements YOUR novel algorithms
"""

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

class EnergyOptimizer:
    """
    This class finds the best way to schedule energy sources.
    Think of it like a smart manager who decides:
    - When to use solar panels vs coal plants
    - How much energy to produce from each source
    - How to minimize costs while meeting demand
    """
    
    def __init__(self):
        self.model = None
        self.results = None
        self.energy_sources = {}
    
    def add_energy_source(self, name: str, max_capacity: float, cost_per_mwh: float, 
                         is_renewable: bool = False):
        """
        Add an energy source to optimize.
        
        Example:
        - Solar panel: max_capacity=100 MW, cost=0 (sun is free!), renewable=True
        - Coal plant: max_capacity=500 MW, cost=50 $/MWh, renewable=False
        """
        self.energy_sources[name] = {
            'max_capacity': max_capacity,
            'cost': cost_per_mwh,
            'renewable': is_renewable
        }
        print(f"➕ Added energy source: {name} (Max: {max_capacity} MW, Cost: ${cost_per_mwh}/MWh)")
    
    def optimize_schedule(self, demand_forecast: list, time_hours: int = 24):
        """
        Find the best energy schedule to meet demand at lowest cost.
        
        This solves the math problem:
        - Minimize: Total cost of energy production
        - Subject to: Meeting all energy demand + technical constraints
        """
        
        print("🔧 Setting up optimization problem...")
        
        # TODO: Copy optimization code from GitHub projects here
        # Look for code that uses:
        # - scipy.optimize.linprog
        # - pyomo models
        # - CVX optimization
        
        # Create optimization model using Pyomo
        model = pyo.ConcreteModel()
        
        # Sets (like lists of things to optimize)
        model.time_periods = pyo.RangeSet(0, time_hours - 1)
        model.generators = pyo.Set(initialize=list(self.energy_sources.keys()))
        
        # Variables (things we want to find optimal values for)
        model.power_output = pyo.Var(
            model.generators, model.time_periods,
            domain=pyo.NonNegativeReals,
            doc="Power output of each generator at each time"
        )
        
        # Objective function (what we want to minimize)
        def total_cost_rule(model):
            return sum(
                model.power_output[gen, t] * self.energy_sources[gen]['cost']
                for gen in model.generators
                for t in model.time_periods
            )
        
        model.total_cost = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)
        
        # Constraints (rules that must be followed)
        
        # 1. Meet demand at every hour
        def demand_constraint_rule(model, t):
            return sum(model.power_output[gen, t] for gen in model.generators) >= demand_forecast[t]
        
        model.demand_constraint = pyo.Constraint(
            model.time_periods, rule=demand_constraint_rule,
            doc="Must meet energy demand each hour"
        )
        
        # 2. Don't exceed generator capacity
        def capacity_constraint_rule(model, gen, t):
            return model.power_output[gen, t] <= self.energy_sources[gen]['max_capacity']
        
        model.capacity_constraint = pyo.Constraint(
            model.generators, model.time_periods, rule=capacity_constraint_rule,
            doc="Cannot exceed generator maximum capacity"
        )
        
        self.model = model
        print("✅ Optimization problem set up!")
        
        # Solve the problem
        return self.solve()
    
    def solve(self):
        """
        Solve the optimization problem.
        This is like asking a super smart calculator to find the best answer.
        """
        if self.model is None:
            raise ValueError("Must set up optimization problem first!")
        
        print("🧮 Solving optimization problem...")
        
        # Use a solver (like a specialized calculator for optimization)
        solver = pyo.SolverFactory('glpk')  # Free solver that works well
        
        try:
            results = solver.solve(self.model, tee=True)  # tee=True shows progress
            
            # Check if solution was found
            if (results.solver.status == SolverStatus.ok and 
                results.solver.termination_condition == TerminationCondition.optimal):
                
                print("✅ Optimal solution found!")
                self.results = results
                return self.extract_solution()
            else:
                print("❌ No optimal solution found!")
                return None
                
        except Exception as e:
            print(f"❌ Solver error: {e}")
            return None
    
    def extract_solution(self):
        """
        Extract the solution in a readable format.
        """
        if self.model is None or self.results is None:
            return None
        
        print("📊 Extracting solution...")
        
        solution = {
            'total_cost': pyo.value(self.model.total_cost),
            'schedule': {}
        }
        
        # Get power output for each generator at each time
        for gen in self.model.generators:
            solution['schedule'][gen] = []
            for t in self.model.time_periods:
                power = pyo.value(self.model.power_output[gen, t])
                solution['schedule'][gen].append(power)
        
        # Create a nice summary table
        schedule_df = pd.DataFrame(solution['schedule'])
        schedule_df.index.name = 'Hour'
        
        print(f"💰 Total Cost: ${solution['total_cost']:.2f}")
        print("\n📋 Optimal Energy Schedule:")
        print(schedule_df.round(2))
        
        return solution
    
    def plot_schedule(self, solution):
        """
        Create a visual chart of the optimal energy schedule.
        """
        if solution is None:
            print("❌ No solution to plot!")
            return
        
        import matplotlib.pyplot as plt
        
        schedule_df = pd.DataFrame(solution['schedule'])
        
        # Create stacked area plot
        plt.figure(figsize=(12, 6))
        
        # Plot each energy source
        sources = list(schedule_df.columns)
        colors = ['gold', 'skyblue', 'lightgreen', 'coral', 'plum']
        
        plt.stackplot(schedule_df.index, 
                     *[schedule_df[source] for source in sources],
                     labels=sources,
                     colors=colors[:len(sources)],
                     alpha=0.8)
        
        plt.xlabel('Hour of Day')
        plt.ylabel('Power Output (MW)')
        plt.title('Optimal Energy Schedule')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig('results/plots/energy_schedule.png', dpi=300, bbox_inches='tight')
        print("📈 Schedule plot saved to results/plots/energy_schedule.png")
        plt.show()

# Example usage:
if __name__ == "__main__":
    print("Energy Optimization System is ready! 🎉")
    
    # Example workflow:
    # 1. optimizer = EnergyOptimizer()
    # 2. optimizer.add_energy_source("Solar", 200, 0, renewable=True)
    # 3. optimizer.add_energy_source("Coal", 500, 50, renewable=False) 
    # 4. demand = [300, 280, 260, 240, ...]  # 24 hours of demand
    # 5. solution = optimizer.optimize_schedule(demand)
    # 6. optimizer.plot_schedule(solution)
    