"""Experiment 3: exact nearest-symbol search costs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_experiment import main
if __name__=='__main__':main('activation_cost')
