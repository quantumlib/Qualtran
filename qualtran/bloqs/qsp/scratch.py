import numpy as np
from scipy.optimize import minimize

# DTYPE = np.complex256
# TOL = 1e-11
class ReflectionConvolver:
    def __init__(self, poly, granularity=None):
        if granularity is None:
            granularity = poly.shape[0]
        P = np.pad(poly, (0, granularity - poly.shape[0]))

        ft = np.fft.fft(P)

        # Normalize P
        P_norms = np.abs(ft)
        nomralized_poly = poly / np.max(P_norms)

        self.conv_p_negative = self.complex_conv_by_flip_conj(nomralized_poly.real, nomralized_poly.imag) * -1
        self.conv_p_negative[nomralized_poly.shape[0] - 1] = 1 - np.linalg.norm(nomralized_poly) ** 2
        self.P = nomralized_poly

    def loss_function(self, x):
        real_part = x[:len(x) // 2]
        imag_part = x[len(x) // 2:]
        conv_result = self.complex_conv_by_flip_conj(real_part, imag_part)

        # Compute loss using squared distance function
        loss = np.linalg.norm(self.conv_p_negative - conv_result) ** 2
        return loss

    def complex_conv_by_flip_conj(self, real_part, imag_part):
        real_flip = np.flip(real_part, axis=[0])
        imag_flip = np.flip(-1 * imag_part, axis=[0])

        conv_real_part = np.convolve(real_part, real_flip, mode="full")
        conv_imag_part = np.convolve(imag_part, imag_flip, mode="full")

        conv_real_imag = np.convolve(real_part, imag_flip, mode="full")
        conv_imag_real = np.convolve(imag_part, real_flip, mode="full")

        # Compute real and imaginary part of the convolution
        real_conv = conv_real_part - conv_imag_part
        imag_conv = conv_real_imag + conv_imag_real

        # Combine to form the complex result
        return real_conv + 1j * imag_conv

def make_q(poly, dtype, tol):

    q_initial = np.random.randn(poly.shape[0]*2).astype(dtype=dtype)
    q_initial_normalized = q_initial / np.linalg.norm(q_initial)

    rf = ReflectionConvolver(poly)

    minimizer = minimize(rf.loss_function,q_initial_normalized, method="L-BFGS-B", tol=tol)
    return rf.P, array_to_complex(minimizer.x)

def array_to_complex(x):
    real_part = x[:len(x) // 2]
    imag_part = x[len(x) // 2:]
    return real_part+ 1.j*imag_part


def run(DTYPE, TOL):
    poly = np.array([1,2,3,4,5], dtype=DTYPE)
    P,Q = make_q(poly, dtype, TOL)
    # print(np.abs(P))
    # print(np.abs(Q))
    # print(Q)
    return -np.log10(abs(1-np.sum(np.abs(Q)**2+np.abs(P)**2)))

dtypes = [np.complex64, np.complex128, np.complex256]
precisions = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-15]
trials = 5

for dtype in dtypes:
    for precision in precisions:
        sum = 0
        for i in range(trials):
            sum += run(dtype, precision)
        print(dtype, precision, sum/trials)

# use complex 128 and tol e-12