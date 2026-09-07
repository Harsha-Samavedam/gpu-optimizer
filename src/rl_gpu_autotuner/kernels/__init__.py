"""GPU kernel implementations.

Modules in this package import Triton lazily through the hardware-backed
benchmark. Keeping them separate lets the rest of the project run on machines
without CUDA or Triton installed.
"""
