# fix small bug in CUTLASS code base
sed -i 's/::hexp(x.to_half());/__half2float(::hexp(x.to_half()));/' ./thirdparty/cutlass/include/cutlass/fast_math.h 

pip uninstall dfss
python -W ignore setup.py build
python -W ignore setup.py install