# VMTransformer 


Train
```
% python training/train.py              
```

Test 
```
python testing/test.py --model /weights/frac0.01_batch128_20250207_191250/model.h5 --output-dir evaluation_results --data-fraction 0.1
```
or
```
python testing/test.py \
    --model weights/frac0.01_batch128_20250207_191250/model.h5 \
    --output-dir evaluation_results \
    --data-fraction 0.005
```

Predict
```
python prediction/predict.py --model weights/frac0.01_batch128_20250207_191250/model.h5
```

