SP=$(python -c "import site;print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$(ls -d $SP/nvidia/*/lib | tr "\n" ":")$LD_LIBRARY_PATH
export PYTHONPATH=/b/pylib_score HOME=/b
cd /b && python score.py 2>&1 | grep -v -i -E "warn|^ *$|deprecat|Applied providers|find model|set det-size|input mean"
