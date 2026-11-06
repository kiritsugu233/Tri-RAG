#!/usr/bin/env python3
"""CPU closed-record attribution; preflight never consumes the unique run."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mechanism import tls_rag_step5_pq_attribution_v1 as a


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['preflight','run'])
    args=p.parse_args();c=a.config()
    source=a.ROOT/c['source_directory']
    if args.action=='preflight':
        a.verify_inputs(c,source);print('Closed arrays verified; no science executed.')
    else:
        a.run(c,source,a.ROOT/c['output_directory'])


if __name__=='__main__':main()
