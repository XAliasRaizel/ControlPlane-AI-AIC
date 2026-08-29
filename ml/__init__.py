"""ControlPlane.ai optional ML training/eval utilities.

Nothing imported at package import time requires torch/transformers/sklearn;
the heavy stack is imported lazily only inside the functions that train or
run models, so `import ml.common` works on the default (GPU-free) install.
"""
