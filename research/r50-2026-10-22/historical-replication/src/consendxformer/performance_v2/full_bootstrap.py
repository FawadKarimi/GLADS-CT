"""Fixed -I source bootstrap; no arbitrary module selection."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from consendxformer.performance_v2.full_worker import main
if __name__=='__main__': sys.exit(main(sys.argv[1]))
