import numpy as np
from scipy.optimize import minimize
class FastQSP:
    def __init__(self, poly):
        self.conv_p_negative = self.complex_conv_by_flip_conj(poly.real, poly.imag) * -1
        self.conv_p_negative[poly.shape[0] - 1] = 1 - np.linalg.norm(poly) ** 2

    def normalize(self, input_poly, granularity=8):
        P = np.pad(input_poly, (0, 2**granularity - input_poly.shape[0]))
        ft = np.fft.fft(P)

        # Normalize P
        P_norms = np.abs(ft)
        return  input_poly / np.max(P_norms)

    def loss_function(self, x):
        real_part = x[:len(x) // 2]
        imag_part = x[len(x) // 2:]
        conv_result = self.complex_conv_by_flip_conj(real_part, imag_part)

        # Compute loss using squared distance function
        loss = np.linalg.norm(self.conv_p_negative - conv_result) ** 2
        return loss
    @staticmethod
    def array_to_complex(x):
        real_part = x[:len(x) // 2]
        imag_part = x[len(x) // 2:]
        return real_part + 1.j * imag_part

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

def fast_complementary_polynomial(poly, verify=True, granularity=8):
    DTYPE = np.complex128
    TOLERANCE = 1e-12
    poly = poly.astype(DTYPE)
    np.random.seed(42)
    q_initial = np.random.randn(poly.shape[0]*2)
    q_initial_normalized = q_initial / np.linalg.norm(q_initial)

    qsp = FastQSP(poly)

    minimizer = minimize(qsp.loss_function,q_initial_normalized, method="L-BFGS-B", tol=TOLERANCE)

    return qsp.array_to_complex(minimizer.x)


