import ipopt
import numpy as np
# np.set_printoptions(threshold=np.nan,linewidth=500,precision=3,suppress=True)

from .constraints.constraintscontainer import cCostraintsContainer
from .constraints.constraints import cAffineBilateralConstraint


class cIpoptInterface(cCostraintsContainer):
    def __init__(self, _cost, _execution_time):
        self.cost_ = _cost
        self.n_ = _cost.wp_.shape[0] - 1
        self.T_ = _execution_time
        super(cIpoptInterface, self).__init__(_n=self.n_)

        self.finiteTimeConstraint_ =\
            cAffineBilateralConstraint(
                A=np.array([[1.0 for i in range(0, self.n_)]]),
                b=np.array([-self.T_]))

        self.append(self.finiteTimeConstraint_)

        self.allocateBuffers()

        self.gradbuff = np.zeros((self.n_, ))

        self.lowerBounds = np.array([0.0 for i in range(self.n_)])

        self.upperBounds = np.array([1.e19 for i in range(self.n_)])

        self.constraintsLowerBounds = \
            np.hstack([c.lowerBounds() for c in self])

        self.constraintsUpperBounds = \
            np.hstack([c.upperBounds() for c in self])

        self.m_ = len(self.constraintsLowerBounds)

        N = _cost.N_
        self.P_ = np.eye(N) - (1.0 / N) * np.ones((N, N))

    def objective(self, x):
        return self.cost_(x)

    def gradient(self, x):
        self.cost_.gradient(x, self.gradbuff)
        return self.gradbuff

    def constraints(self, x):
        return self.value(x)

    def solve(self):
        nlp = ipopt.problem(
            n=self.n_,
            m=self.m_,
            problem_obj=self,
            lb=self.lowerBounds,
            ub=self.upperBounds,
            cl=self.constraintsLowerBounds,
            cu=self.constraintsUpperBounds)

        nlp.addOption(b'print_level', 5)
        nlp.addOption(b'file_print_level', 12)
        nlp.addOption(b'output_file', b'ipopt_logs.txt')
        # nlp.addOption(b'derivative_test_tol',5.0e-5)
        # nlp.addOption(b'derivative_test', b'first-order')
        # nlp.addOption(b'derivative_test_perturbation', 1.0e-6)
        nlp.addOption(b'tol', 5.0e-6)
        nlp.addOption(b'max_iter', 500)
        nlp.addOption(b'hessian_approximation', b'limited-memory')
        nlp.addOption(b'jac_c_constant', b'yes')

        x0 = self.cost_.get_first_guess()
        return nlp.solve(x0)


