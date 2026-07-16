import qualtran as qlt
import qualtran.dtype as qdt

from rsqualtran import QLTFastsim

from qualtran.bloqs.arithmetic import Negate


def main():
    bloq = Negate(qdt.QInt(8))
    simulator = QLTFastsim.from_bloq(bloq)
    print(simulator)
    for x in range(2**7):
        (result_x,) = simulator.call_classically(x=x)
        print(f"x={x} -> result_x={result_x}")


if __name__ == "__main__":
    main()
