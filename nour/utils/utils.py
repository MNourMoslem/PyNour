
class GradientFunction:
    _gradeint_functions_ = {}

    def __init__(self, fun_name):
        self.op = fun_name
        
    def __call__(self, f):
        GradientFunction._gradeint_functions_[self.op] = f
        return f
    
class OptimizerClass:
    _optimizer_classes_ = {}

    def __init__(self, optimizer_name):
        self.optimizer_name = optimizer_name
        
    def __call__(self, optim):
        OptimizerClass._optimizer_classes_[self.optimizer_name] = optim
        return optim
    
class LossFunction:
    _loss_functions_ = {}

    def __init__(self, loss_fun_name):
        self.loss_fun_name = loss_fun_name
    
    def __call__(self, loss_fn):
        LossFunction._loss_functions_[self.loss_fun_name] = loss_fn
        return loss_fn
        
class no_grad:
    _grad_mode_ = True

    def __enter__(self, *args):
        no_grad._grad_mode_ = False
        
    def __exit__(self, *args):
        no_grad._grad_mode_ = True
