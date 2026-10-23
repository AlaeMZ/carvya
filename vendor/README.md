# vendor/

`libgomp.so.1` (the GNU OpenMP runtime, taken from Amazon Linux 2023) is committed because LightGBM needs it and Vercel's Python runtime does not ship it. `src/models/predict.py` loads this copy only when the system has no libgomp of its own, so local and Docker runs use the system library.
