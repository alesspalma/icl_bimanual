# Debug file, debug.py
from RLBench.tools.dataset_generator_bimanual import main
import debugpy

def helper() -> None:
    print("Waiting for debugger attach...")
    debugpy.listen(5678)
    debugpy.wait_for_client()
    main() # we call the main by passing the config given by command line

if __name__ == '__main__':
    helper()