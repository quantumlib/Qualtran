import base64
import gzip

import qualtran as qlt


def qlt_to_url(code: str) -> str:
    return base64.urlsafe_b64encode(gzip.compress(code.encode('utf-8'))).decode('ascii').rstrip('=')


def bloq_to_url(bloq: qlt.Bloq, base: str = 'http://localhost:5173') -> str:
    from qualtran.l1 import dump_root_l1

    return f"{base}/qlt/{qlt_to_url(dump_root_l1(bloq))}"
